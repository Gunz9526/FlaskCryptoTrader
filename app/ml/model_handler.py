import pandas as pd
import joblib
import numpy as np
from typing import Literal, Dict, Any
import logging
import os

MODEL_BASE_PATH = "./models" 

class ModelHandler:
    def __init__(self, symbol: str, model_type: Literal['ranging', 'trending']):
        self.models: Dict[str, Dict[str, Any]] = {}
        symbol_path = symbol.replace('/', '_').lower()
        model_dir = f"{MODEL_BASE_PATH}/{symbol_path}"
        
        model_names = ['lgbm', 'xgb', 'catboost']
        for name in model_names:
            path = f"{model_dir}/{model_type}_{name}_artifact.pkl"
            try:
                artifacts = joblib.load(path)
                self.models[name] = artifacts
                logging.info(f"Loaded '{name}' artifact for {symbol}/{model_type}")
            except FileNotFoundError:
                logging.warning(f"Artifact for '{name}' not found at {path}")
            except Exception as e:
                logging.error(f"Error loading artifact for '{name}': {e}", exc_info=True)

    def _prepare_features(self, features: pd.DataFrame, model_name: str, model_artifacts: Dict[str, Any]) -> pd.DataFrame | None:
        try:
            features_copy = features.copy()

            if model_name == 'catboost':
                if 'rsi' in features_copy.columns:
                    features_copy['rsi_binned'] = pd.cut(
                        features_copy['rsi'],
                        bins=[0, 30, 70, 100],
                        labels=['low', 'mid', 'high'],
                        include_lowest=True
                    ).astype(str).fillna('missing')
                else:
                    features_copy['rsi_binned'] = 'missing'

            scaler = model_artifacts['scaler']
            features_order = model_artifacts['features']
            
            aligned_features = features_copy.reindex(columns=features_order, fill_value=0)

            numeric_features = aligned_features.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_features:
                aligned_features[numeric_features] = scaler.transform(aligned_features[numeric_features])
            
            return aligned_features

        except Exception as e:
            logging.error(f"Error preparing features for {model_name}: {e}", exc_info=True)
            return None
        
    def _get_ensemble_probas(self, features: pd.DataFrame) -> tuple[np.ndarray | None, list]:
        if not self.models:
            return None, []

        all_probas_list = []
        model_names = []
        for name, artifacts in self.models.items():
            scaled_features = self._prepare_features(features, name, artifacts)
            if scaled_features is not None:
                try:
                    probas = artifacts['model'].predict_proba(scaled_features)
                    if probas.shape[0] > 0 and probas.shape[1] == 3:
                        all_probas_list.append(probas[0]) 
                        model_names.append(name)
                    else:
                        logging.warning(f"Model '{name}' did not return valid probabilities. Shape: {probas.shape}")
                except Exception as e:
                    logging.error(f"Error getting prediction from {name}: {e}", exc_info=True)

        if not all_probas_list:
            return None, []

        ensembled_probas = np.mean(np.array(all_probas_list), axis=0)
        return ensembled_probas, model_names


    def get_ensemble_probas_for_df(self, features_df: pd.DataFrame) -> np.ndarray | None:
        if not self.models:
            return None

        weighted_probas = []
        weights = []

        for name, artifacts in self.models.items():
            estimator = artifacts.get('calibrated_model') or artifacts.get('model')
            if estimator is None:
                logging.warning(f"No estimator found for model '{name}'")
                continue

            scaled_features = self._prepare_features(features_df, name, artifacts)
            if scaled_features is None:
                continue

            try:
                probas = estimator.predict_proba(scaled_features)
                if probas.shape[1] != 3:
                    logging.warning(f"Model '{name}' did not return 3-class probabilities. Shape: {probas.shape}")
                    continue
                w = max(float(artifacts.get('f1_score', 0.5)), 1e-6)
                weighted_probas.append(probas)
                weights.append(w)
            except Exception as e:
                logging.error(f"Error getting predictions from {name}: {e}", exc_info=True)

        if not weighted_probas:
            return None

        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()

        stacked = np.stack(weighted_probas, axis=0)
        ensembled = np.tensordot(w_arr, stacked, axes=(0, 0))
        logging.info(f"Ensemble weights: {[round(w,3) for w in w_arr.tolist()]}")
        return ensembled

    def get_prediction_confidence(self, features: pd.DataFrame) -> float:
        probas = self.get_ensemble_probas_for_df(features)
        if probas is None or probas.shape[0] == 0:
            return 0.0
        return float(np.max(probas[0]))


    def get_ensemble_probas_for_df(self, features_df: pd.DataFrame) -> np.ndarray | None:
        if not self.models:
            return None

        all_probas_list = []
        for name, artifacts in self.models.items():
            model = artifacts['model']
            scaled_features = self._prepare_features(features_df, name, artifacts)
            if scaled_features is not None:
                try:
                    probas = model.predict_proba(scaled_features)
                    if probas.shape[1] == 3:
                        all_probas_list.append(probas)
                    else:
                        logging.warning(f"Model '{name}' did not return 3-class probabilities. Shape: {probas.shape}")
                except Exception as e:
                    logging.error(f"Error getting predictions from {name}: {e}", exc_info=True)

        if not all_probas_list:
            return None

        ensembled_probas = np.mean(np.stack(all_probas_list), axis=0)
        return ensembled_probas
    
    def get_ensemble_prediction(self, features: pd.DataFrame) -> int:
        probas = self.get_ensemble_probas_for_df(features)
        if probas is None or probas.shape[0] == 0:
            logging.warning("No probabilities from ensemble. Returning neutral signal 0.")
            return 0
        idx = int(np.argmax(probas[0]))
        mapping = {0: -1, 1: 0, 2: 1}
        return mapping.get(idx, 0)