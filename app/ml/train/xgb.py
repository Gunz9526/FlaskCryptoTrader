import os
import pandas as pd
import xgboost as xgb
import optuna
from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import f1_score
from .base import BaseTreeTrainer

class XGBTrainer(BaseTreeTrainer):
    @property
    def model_name(self) -> str:
        return "xgb"

    def _prepare_model_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict, *, sample_weight=None, class_weight_map=None):
        final_params = {
            'objective': 'multi:softprob', 'eval_metric': 'merror', 'num_class': 3,
            'seed': 42, 'n_jobs': -1, **params
        }
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    def _get_objective_func(self, X_train, y_train, X_val, y_val):
        def objective(trial: optuna.Trial):
            params = {
                'objective': 'multi:softprob', 'eval_metric': 'merror', 'num_class': 3,
                'booster': 'gbtree', 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-merror")
            model = xgb.XGBClassifier(callbacks=[pruning_callback], **params)

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val, iteration_range=(0, model.best_iteration))
            return f1_score(y_val, preds, average='weighted')
        return objective