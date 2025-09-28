import logging
from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

from app.trading.feature_engineering import add_technical_indicators, create_advanced_features, prepare_features_for_model, create_labels

class BaseModelTrainer(ABC):
    
    def __init__(self, symbol: str, model_type: str):
        self.symbol = symbol
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.features_order = None

    @abstractmethod
    def train(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame):
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class BaseTreeTrainer(BaseModelTrainer):
    def __init__(self, symbol: str, model_type: str):
        super().__init__(symbol, model_type)
        self.n_splits = 5

    def _prepare_base_features_and_labels(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features_df = add_technical_indicators(df_15m, df_1h)
        features_df = create_advanced_features(features_df)
        
        y = create_labels(features_df, self.model_type)
        X = prepare_features_for_model(features_df)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        combined = X.join(y.rename('label')).dropna()
        
        X = combined.drop(columns=['label'])
        y = combined['label'].map({-1: 0, 0: 1, 1: 2})
        
        return X, y

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        from sklearn.model_selection import train_test_split

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        X_train_full, X_test, y_train_full, y_test = None, None, None, None
        for train_index, test_index in tscv.split(X):
            X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]
            
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        self.scaler = StandardScaler()
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        
        X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()
        
        if numeric_features:
            X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
            X_val_scaled[numeric_features] = self.scaler.transform(X_val[numeric_features])
            X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
            
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    
    def _compute_class_weights(self, y: pd.Series) -> tuple[dict[int, float], np.ndarray]:
        classes = np.array(sorted(pd.unique(y)))
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
        sample_weight = y.map(class_weight_map).astype(float).values
        return class_weight_map, sample_weight


    def _save_artifacts(self, model, calibrated_model, f1: float, trained_until, scaler: StandardScaler, features: list[str]):
        symbol_path = self.symbol.replace('/', '_').lower()
        model_dir = os.path.join("models", symbol_path)
        os.makedirs(model_dir, exist_ok=True)

        filename = f"{self.model_type}_{self.model_name}_artifact.pkl"
        path = os.path.join(model_dir, filename)

        joblib.dump({
            "model": model,
            "calibrated_model": calibrated_model,
            "scaler": scaler,
            "features": features,
            "f1_score": f1,
            "trained_until": trained_until
        }, path)
        logging.info(f"[{self.model_name}] Artifacts saved to {path}")


    
    def train(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame):
        try:
            X, y = self._prepare_base_features_and_labels(df_15m, df_1h)

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            train_index, test_index = list(tscv.split(X))[-1]
            X_trainval, X_test = X.iloc[train_index], X.iloc[test_index]
            y_trainval, y_test = y.iloc[train_index], y.iloc[test_index]

            tscv_inner = TimeSeriesSplit(n_splits=3)
            inner_train_idx, inner_val_idx = list(tscv_inner.split(X_trainval))[-1]
            X_train, X_val = X_trainval.iloc[inner_train_idx], X_trainval.iloc[inner_val_idx]
            y_train, y_val = y_trainval.iloc[inner_train_idx], y_trainval.iloc[inner_val_idx]

            X_train = self._prepare_model_specific_features(X_train)
            X_val = self._prepare_model_specific_features(X_val)
            X_test = self._prepare_model_specific_features(X_test)
            X_train_scaled, X_val_scaled, X_test_scaled = self._scale_features(X_train, X_val, X_test)
            self.features_order = X_train_scaled.columns.tolist()

            logging.info(f"[{self.model_name}] Tuning hyperparameters...")
            best_params = self._tune_hyperparams(X_train_scaled, y_train, X_val_scaled, y_val)
            logging.info(f"[{self.model_name}] Best params found: {best_params}")

            X_train_full = pd.concat([X_train_scaled, X_val_scaled])
            y_train_full = pd.concat([y_train, y_val])

            class_weight_map, sample_weight_full = self._compute_class_weights(y_train_full)

            logging.info(f"[{self.model_name}] Training final model with class/sample weights...")
            base_model = self._train_model(
                X_train_full, y_train_full, best_params,
                sample_weight=sample_weight_full,
                class_weight_map=class_weight_map
            )

            logging.info(f"[{self.model_name}] Fitting calibration model (isotonic, cv='prefit') on validation split...")
            calibrator = CalibratedClassifierCV(estimator=base_model, cv='prefit', method='isotonic')
            calibrator.fit(X_val_scaled, y_val)

            y_pred = calibrator.predict(X_test_scaled)
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            f1 = f1_score(y_test, y_pred, average='weighted')
            logging.info(f"[{self.model_name}] Final calibrated F1 on test set: {f1:.4f}")

            self._save_artifacts(base_model, calibrator, f1, X_train_full.index[-1], self.scaler, self.features_order)
            logging.info(f"[{self.model_name}] Training completed and artifacts saved.")
        except Exception as e:
            logging.error(f"Failed to train {self.model_name} for {self.symbol}/{self.model_type}: {e}", exc_info=True)



    @abstractmethod
    def _prepare_model_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _tune_hyperparams(self, X_train, y_train, X_val, y_val) -> dict:
        import optuna

        def objective(trial: optuna.Trial):
            raise NotImplementedError("Objective function must be implemented in child class.")

        objective = self._get_objective_func(X_train, y_train, X_val, y_val)

        storage_dir = "optuna_tuning"
        os.makedirs(storage_dir, exist_ok=True)
        storage_name = f"sqlite:///{storage_dir}/{self.symbol.replace('/', '_')}_{self.model_type}_{self.model_name}.db"
        
        study = optuna.create_study(
            direction='maximize',
            storage=storage_name,
            study_name=f"{self.model_name}_study",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=30, n_jobs=1) 
        
        return study.best_params

    @abstractmethod
    def _get_objective_func(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict):
        raise NotImplementedError