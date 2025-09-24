import os
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
from optuna_integration import LightGBMPruningCallback
from sklearn.metrics import f1_score
from .base import BaseTreeTrainer

class LGBMTrainer(BaseTreeTrainer):
    @property
    def model_name(self) -> str:
        return "lgbm"
    
    def _prepare_model_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict):
        final_params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3,
            'verbose': -1, 'n_jobs': -1, 'seed': 42, **params
        }
        model = lgb.LGBMClassifier(**final_params)
        model.fit(X_train, y_train)
        return model

    def _get_objective_func(self, X_train, y_train, X_val, y_val):
        
        def lgb_f1_score(y_true, y_pred):
            y_pred_labels = np.argmax(y_pred.reshape(len(np.unique(y_true)), -1), axis=0)
            f1 = f1_score(y_true, y_pred_labels, average='weighted')
            return 'weighted_f1', f1, True

        def objective(trial):
            params = {
                'objective': 'multiclass','metric': 'multi_logloss', 'num_class': 3, 'random_state': 42,
                'verbose': -1,'random_state': 42, 'verbose': -1,
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            model = lgb.LGBMClassifier(**params)
            
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, 
                "weighted_f1",
                valid_name="valid_0"
            )
            
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric=lgb_f1_score,
                      callbacks=[pruning_callback, lgb.early_stopping(50, verbose=False)])
            
            preds = model.predict(X_val, num_iteration=model.best_iteration_)
            return f1_score(y_val, preds, average='weighted')
            
        return objective