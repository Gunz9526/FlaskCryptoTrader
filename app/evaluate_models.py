import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import text


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


OUTPUT_DIR = "test_result"
TRANSACTION_COST = 0.001
SUPPORTED_SYMBOLS = ['BTC/USD', 'ETH/USD']
MODEL_TYPES = ['ranging', 'trending']
MODEL_NAMES = ['lgbm', 'xgb', 'catboost']


from app.factory import create_app, db
from app.ml.model_handler import ModelHandler
from app.ml.train.lgbm import LGBMTrainer
from app.ml.train.xgb import XGBTrainer
from app.ml.train.catboost import CatBoostTrainer



def run_backtest(df_price: pd.DataFrame, signals: pd.Series) -> dict:
    df = pd.DataFrame(index=df_price.index)
    df['close'] = df_price['close']
    df['signal'] = signals.reindex(df.index).fillna(0)
    df['market_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['strategy_returns'] = df['market_returns'] * df['signal'].shift(1)
    
    trades = df['signal'].diff().ne(0)
    df.loc[trades, 'strategy_returns'] -= TRANSACTION_COST
    df.dropna(inplace=True)

    if df.empty or df['strategy_returns'].std() == 0:
        return {"sharpe_ratio": 0, "max_drawdown": 0, "cumulative_returns": pd.Series([1.0], index=[df_price.index[0]])}

    daily_returns = df['strategy_returns'].resample('D').sum()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() != 0 else 0

    cumulative_returns = (1 + df['strategy_returns']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    
    return {"sharpe_ratio": sharpe_ratio, "max_drawdown": drawdown.min(), "cumulative_returns": cumulative_returns}

def plot_confusion_matrix(y_true, y_pred, model_name, symbol, model_type):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Neutral', 'Buy'], yticklabels=['Sell', 'Neutral', 'Buy'])
    plt.title(f'Confusion Matrix: {model_name} ({symbol}/{model_type})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plot_path = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}_{model_type}_{model_name}_cm.png")
    plt.savefig(plot_path)
    plt.close()


def run_evaluation_pipeline(symbol: str, model_type: str):
    logging.info(f"===== Running evaluation for {symbol}/{model_type} =====")
    
    app = create_app()
    with app.app_context():
        stmt_15m = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '15m' ORDER BY timestamp ASC")
        stmt_1h = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '1h' ORDER BY timestamp ASC")
        df_15m = pd.read_sql(stmt_15m, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')
        df_1h = pd.read_sql(stmt_1h, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')

    if len(df_15m) < 2000:
        logging.error(f"Not enough data for {symbol} to run evaluation. Skipping.")
        return

    data_preparer = LGBMTrainer(symbol, model_type)
    X_raw, y_raw = data_preparer._prepare_base_features_and_labels(df_15m, df_1h)
    
    tscv = TimeSeriesSplit(n_splits=5)
    test_indices = list(tscv.split(X_raw))[-1]
    X_test_raw, y_test = X_raw.iloc[test_indices], y_raw.iloc[test_indices]
    
    price_df_test = df_15m.loc[X_test_raw.index]
    
    results = []
    symbol_path = symbol.replace('/', '_').lower()
    model_dir = f"./models/{symbol_path}"

    for name in MODEL_NAMES:
        try:
            path = f"{model_dir}/{model_type}_{name}_artifact.pkl"
            artifacts = joblib.load(path)
            
            if name == 'lgbm': trainer_instance = LGBMTrainer(symbol, model_type)
            elif name == 'xgb': trainer_instance = XGBTrainer(symbol, model_type)
            else: trainer_instance = CatBoostTrainer(symbol, model_type)

            X_test_specific = trainer_instance._prepare_model_specific_features(X_test_raw)
            
            scaler = artifacts['scaler']
            numeric_features = X_test_specific.select_dtypes(include=np.number).columns.tolist()
            X_test_scaled = X_test_specific.copy()
            if numeric_features:
                X_test_scaled[numeric_features] = scaler.transform(X_test_specific[numeric_features])
            
            X_test_final = X_test_scaled.reindex(columns=artifacts['features'], fill_value=0)

            model = artifacts['model']
            y_pred = model.predict(X_test_final)

            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            report = classification_report(y_test, y_pred, target_names=['Sell', 'Neutral', 'Buy'], output_dict=True, zero_division=0)
            plot_confusion_matrix(y_test, y_pred, name, symbol, model_type)
            
            signals_for_backtest = pd.Series(y_pred, index=y_test.index).map({0: -1, 1: 0, 2: 1})
            backtest_results = run_backtest(price_df_test, signals_for_backtest)
            
            results.append({
                "model": name, "f1_score": report['weighted avg']['f1-score'], 
                "precision_buy": report['Buy']['precision'], "recall_buy": report['Buy']['recall'],
                "precision_sell": report['Sell']['precision'], "recall_sell": report['Sell']['recall'],
                **backtest_results
            })
            logging.info(f"Evaluation completed for single model: {name}")

        except FileNotFoundError:
            logging.warning(f"Artifact for '{name}' not found. Skipping.")
        except Exception as e:
            logging.error(f"Error evaluating single model {name}: {e}", exc_info=True)

    try:
        handler = ModelHandler(symbol, model_type)
        if handler.models:
            ensemble_probas = handler.get_ensemble_probas_for_df(X_test_raw)
            if ensemble_probas is not None:
                y_pred_ensemble_indices = np.argmax(ensemble_probas, axis=1)
                y_pred_ensemble = pd.Series(y_pred_ensemble_indices, index=y_test.index)
                
                report_ensemble = classification_report(y_test, y_pred_ensemble, target_names=['Sell', 'Neutral', 'Buy'], output_dict=True, zero_division=0)
                plot_confusion_matrix(y_test, y_pred_ensemble, 'ensemble', symbol, model_type)
                
                ensemble_signals = y_pred_ensemble.map({0: -1, 1: 0, 2: 1})
                backtest_ensemble = run_backtest(price_df_test, ensemble_signals)
                
                results.append({
                    "model": "ensemble", "f1_score": report_ensemble['weighted avg']['f1-score'],
                    "precision_buy": report_ensemble['Buy']['precision'], "recall_buy": report_ensemble['Buy']['recall'],
                    "precision_sell": report_ensemble['Sell']['precision'], "recall_sell": report_ensemble['Sell']['recall'],
                    **backtest_ensemble
                })
                logging.info("Evaluation completed for ensemble model.")
    except Exception as e:
        logging.error(f"Ensemble evaluation failed: {e}", exc_info=True)

    if not results:
        logging.error("No models were evaluated.")
        return

    report_df = pd.DataFrame(results).set_index('model')
    report_columns = ['f1_score', 'sharpe_ratio', 'max_drawdown', 'precision_buy', 'recall_buy', 'precision_sell', 'recall_sell']
    report_df = report_df[report_columns]
    
    logging.info(f"\n===== Evaluation Report for {symbol}/{model_type} =====\n" + report_df.to_string(float_format="%.4f"))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f"{symbol_path}_{model_type}_report.md")
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Report for {symbol} ({model_type})\n\n")
        f.write(report_df.to_markdown(floatfmt=".4f"))

    plt.figure(figsize=(14, 7))
    for res in results:
        if 'cumulative_returns' in res and not res['cumulative_returns'].empty:
            res['cumulative_returns'].plot(label=res['model'])
    plt.title(f'Cumulative Returns Comparison: {symbol} ({model_type})')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, f"{symbol_path}_{model_type}_returns.png")
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"Evaluation report and plots saved to '{OUTPUT_DIR}'")

if __name__ == '__main__':
    for symbol in SUPPORTED_SYMBOLS:
        for model_type in MODEL_TYPES:
            run_evaluation_pipeline(symbol, model_type)
