import os
import pandas as pd
import optuna
from optuna_integration import CatBoostPruningCallback
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from .base import BaseTreeTrainer

class CatBoostTrainer(BaseTreeTrainer):
    def __init__(self, symbol: str, model_type: str):
        super().__init__(symbol, model_type)
        self.categorical_features_names = []

    @property
    def model_name(self) -> str:
        return "catboost"

    def _prepare_model_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if 'rsi' in X_copy.columns:
            X_copy['rsi_binned'] = pd.cut(X_copy['rsi'], 
                                          bins=[0, 30, 70, 100], 
                                          labels=['low', 'mid', 'high'], 
                                          include_lowest=True).astype(str)
            X_copy['rsi_binned'] = X_copy['rsi_binned'].fillna('missing')
            self.categorical_features_names = ['rsi_binned']
        return X_copy

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict):
        final_params = {
            'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
            'random_seed': 42, 'verbose': 0,
            'allow_writing_files': False,
            **params
        }
        model = CatBoostClassifier(**final_params)
        model.fit(X_train, y_train, cat_features=self.categorical_features_names)
        return model
    
    def _get_objective_func(self, X_train, y_train, X_val, y_val):
        
        categorical_features_indices = [
            i for i, col in enumerate(X_train.columns) 
            if col in self.categorical_features_names
        ]

        def objective(trial: optuna.Trial):
            params = {
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'random_seed': 42,
                'verbose': 0,
                'allow_writing_files': False,
                'iterations': trial.suggest_int('iterations', 500, 2500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            }
            model = CatBoostClassifier(**params)
            
            pruning_callback = CatBoostPruningCallback(trial, "MultiClass")
            
            model.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      cat_features=categorical_features_indices,
                      early_stopping_rounds=50,
                      callbacks=[pruning_callback],
                      verbose=False)
            
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average='weighted')
        return objective