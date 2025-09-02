import pandas as pd
import logging
import joblib
import numpy as np
from typing import Literal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from .celery_app import celery_instance
from sqlalchemy import select, text
from app.models import MarketData
from app.extensions import db
from app.tasks import SUPPORTED_SYMBOLS
from app.trading.feature_engineering import add_technical_indicators, prepare_features_for_model, create_advanced_features
import os


def create_labels(df: pd.DataFrame, regime: str) -> pd.Series:
    future_periods = 16
    threshold = 0.012
    
    future_returns = df['close'].shift(-future_periods) / df['close'] - 1
    
    if regime == 'trending':
        labels = pd.Series(0, index=df.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
    else:
        labels = pd.Series(0, index=df.index)
        bb_oversold = df['bb_position'] < 0.2
        bb_overbought = df['bb_position'] > 0.8
        rsi_oversold = df['rsi'] < 35
        rsi_overbought = df['rsi'] > 65
        
        labels[(bb_oversold | rsi_oversold) & (future_returns > threshold/2)] = 1
        labels[(bb_overbought | rsi_overbought) & (future_returns < -threshold/2)] = -1
    
    return labels

def train_and_evaluate(symbol: str, model_type: Literal['ranging', 'trending']):
    logging.info(f"Starting training for {model_type} model")
    
    try:
        stmt_15m = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '15m' ORDER BY timestamp ASC")
        stmt_1h = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '1h' ORDER BY timestamp ASC")
        
        df_15m = pd.read_sql(stmt_15m, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')
        df_1h = pd.read_sql(stmt_1h, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')
        
        if len(df_15m) < 2000 or len(df_1h) < 500:
            logging.warning(f"[{symbol}] Insufficient data: 15m={len(df_15m)}, 1h={len(df_1h)}")
            return

        df_features = add_technical_indicators(df_15m, df_1h)
        logging.info(f"[{symbol}] Features added: {len(df_features)} rows, {len(df_features.columns)} columns")
        
        df_features = create_advanced_features(df_features)
        logging.info(f"[{symbol}] Advanced features added: {len(df_features)} rows")
        
        y = create_labels(df_features, model_type)
        logging.info(f"[{symbol}] Labels created: {y.value_counts().to_dict()}")
        
        X = prepare_features_for_model(df_features)
        logging.info(f"[{symbol}] Features prepared: {len(X)} rows, {len(X.columns)} columns")
        
        combined_data = X.join(y.rename('label'))

        combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined_data.dropna(inplace=True)
        if combined_data.empty:
            logging.warning(f"[{symbol}/{model_type}] No valid samples after processing and joining labels.")
            return

        X = combined_data.drop(columns=['label'])
        y = combined_data['label']

        y_raw = y.copy()
        y = y.map({-1: 0, 0: 1, 1: 2})

        if len(X) < 1000:
            logging.warning(f"Too few samples after processing: {len(X)}")
            return

        neutral_ratio = (y_raw == 0).sum() / len(y_raw)
        if neutral_ratio > 0.9:
            logging.warning(f"Too many neutral labels ({neutral_ratio:.2%}) for {model_type}")
            return
        
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, test_idx = list(TimeSeriesSplit(n_splits=5).split(X))[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        classes, counts = np.unique(y_train, return_counts=True)
        class_weights = {c: (len(y_train) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        sample_weight = y_train.map(class_weights)

        lgb_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbose': -1,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_samples': 30
        }
        
        xgb_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 0,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train_scaled, y_train)
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        
        lgb_pred = lgb_model.predict(X_test_scaled)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        lgb_f1 = f1_score(y_test, lgb_pred, average='weighted')
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
        
        logging.info(f"[{symbol}/{model_type}] LightGBM F1: {lgb_f1:.4f}, XGBoost F1: {xgb_f1:.4f}")
        
        model_dir = f"./models/{symbol.replace('/', '_').lower()}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_artifacts = {
            'lgb_model': lgb_model,
            'xgb_model': xgb_model,
            'scaler': scaler,
            'features': list(X_train.columns),
            'metadata': {
                'symbol': symbol,
                'model_type': model_type,
                'n_samples': int(len(X)),
                'train_end': str(X_train.index.max()),
                'test_start': str(X_test.index.min())
            }
        }
        
        try:
            old_artifacts = joblib.load(f'{model_dir}/{model_type}_artifacts.pkl')
            old_lgb_model = old_artifacts['lgb_model']
            old_pred = old_lgb_model.predict(X_test_scaled[old_artifacts['features']])
            old_f1 = f1_score(y_test, old_pred, average='weighted')
            logging.info(f"[{symbol}/{model_type}] Old model F1: {old_f1:.4f}")
        except FileNotFoundError:
            old_f1 = 0.0
        
        if lgb_f1 > old_f1 + 0.015:
            joblib.dump(model_artifacts, f'{model_dir}/{model_type}_artifacts.pkl')
            logging.info(f"[{symbol}/{model_type}] New models and artifacts saved (F1 improvement: {lgb_f1 - old_f1:.4f})")
        else:
            logging.warning(f"[{symbol}/{model_type}] No significant improvement, keeping old models.")

    except Exception as e:
        logging.error(f"Error training {model_type} model for {symbol}: {e}", exc_info=True)

@celery_instance.task(name="tasks.trigger_model_retraining")
def trigger_model_retraining():
    logging.info("Starting comprehensive model retraining cycle")
    
    try:
        for symbol in SUPPORTED_SYMBOLS:
            train_and_evaluate(symbol, 'ranging')
            train_and_evaluate(symbol, 'trending')
        logging.info("Model retraining cycle completed successfully")
    except Exception as e:
        logging.error(f"Error in model retraining: {e}", exc_info=True)