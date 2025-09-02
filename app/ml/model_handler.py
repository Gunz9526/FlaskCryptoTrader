import pandas as pd
import joblib
import numpy as np
from typing import Literal
from sklearn.preprocessing import StandardScaler
import logging
import os

MODEL_BASE_PATH = "./models" 

class ModelHandler:
    def __init__(self, symbol: str, model_type: Literal['ranging', 'trending']):
        self.model_type = model_type
        symbol_path = symbol.replace('/', '_').lower()
        model_dir = f"{MODEL_BASE_PATH}/{symbol_path}"
        artifacts_path = f"{model_dir}/{model_type}_artifacts.pkl"
        
        self.lgb_model = None
        self.xgb_model = None
        self.scaler = None
        self.features_order = None
        
        logging.info(f"Loading '{model_type}' model artifacts for symbol '{symbol}' from {artifacts_path}...")
        try:
            artifacts = joblib.load(artifacts_path)
            self.lgb_model = artifacts['lgb_model']
            self.xgb_model = artifacts['xgb_model']
            self.scaler = artifacts['scaler']
            self.features_order = artifacts['features']
            logging.info(f"Artifacts loaded successfully for {symbol}/{model_type}")
        except FileNotFoundError:
            logging.error(f"Model artifacts not found for '{symbol}/{model_type}'. The model needs to be trained first.")
        except Exception as e:
            logging.error(f"Error loading artifacts for '{symbol}/{model_type}': {e}", exc_info=True)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame | None:
        if self.features_order is None or self.scaler is None:
            return None
        try:
            aligned = features.reindex(columns=self.features_order, fill_value=0.0)
            scaled_np = self.scaler.transform(aligned)
            return pd.DataFrame(scaled_np, index=features.index, columns=self.features_order)
        except Exception as e:
            logging.error(f"Error preparing features for prediction: {e}", exc_info=True)
            return None

    def _decode_label(self, cls: int) -> int:
        return {0: -1, 1: 0, 2: 1}.get(int(cls), 0)

    def get_ensemble_prediction(self, features: pd.DataFrame) -> int:
        if self.lgb_model is None or self.xgb_model is None:
            logging.warning("Models not loaded, returning neutral signal")
            return 0
        features_scaled = self._prepare_features(features)
        if features_scaled is None:
            return 0
        try:
            lgb_proba = self.lgb_model.predict_proba(features_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(features_scaled)[0]

            lgb_pred_class = int(np.argmax(lgb_proba))
            xgb_pred_class = int(np.argmax(xgb_proba))
            lgb_signal = self._decode_label(lgb_pred_class)
            xgb_signal = self._decode_label(xgb_pred_class)
            
            logging.info(
                f"Individual Model Predictions: "
                f"LGBM -> signal={lgb_signal}, confidence={np.max(lgb_proba):.3f} | "
                f"XGB -> signal={xgb_signal}, confidence={np.max(xgb_proba):.3f}"
            )

            avg_proba = (np.array(lgb_proba) + np.array(xgb_proba)) / 2
            cls = int(np.argmax(avg_proba))
            primary = self._decode_label(cls)

            hard_lgb = self._decode_label(int(self.lgb_model.predict(features_scaled)[0]))
            hard_xgb = self._decode_label(int(self.xgb_model.predict(features_scaled)[0]))
            if hard_lgb == hard_xgb and hard_lgb != 0:
                return hard_lgb
            if avg_proba[cls] >= 0.5 and primary != 0:
                return primary
            return 0
        except Exception as e:
            logging.error(f"Error in ensemble prediction: {e}", exc_info=True)
            return 0

    def get_prediction_confidence(self, features: pd.DataFrame) -> float:
        if self.lgb_model is None or self.xgb_model is None:
            return 0.0
        features_scaled = self._prepare_features(features)
        if features_scaled is None:
            return 0.0
        try:
            lgb_proba = self.lgb_model.predict_proba(features_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(features_scaled)[0]
            avg_proba = (np.array(lgb_proba) + np.array(xgb_proba)) / 2
            return float(np.max(avg_proba))
        except Exception as e:
            logging.error(f"Error calculating confidence: {e}", exc_info=True)
            return 0.0