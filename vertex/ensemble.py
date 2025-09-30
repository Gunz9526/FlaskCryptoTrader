import torch
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import List, Dict

class EnsembleModel:
    def __init__(self, model_dir: Path, model_list: List[str], timeframe: str, device: str = 'cpu'):
        self.models = {}
        self.scalers = {}
        self.model_list = model_list
        self.device = device
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime이 설치되어야 합니다. `pip install onnxruntime`")

        for model_type in model_list:
            onnx_path = model_dir / f"{model_type}_{timeframe}_model.onnx"
            scaler_path = model_dir / f"{model_type}_{timeframe}_scaler.pkl"
            
            if not onnx_path.exists() or not scaler_path.exists():
                raise FileNotFoundError(f"'{model_type}' 모델({timeframe})의 ONNX 또는 스케일러 파일을 찾을 수 없습니다: {onnx_path}")

            self.models[model_type] = ort.InferenceSession(str(onnx_path))
            self.scalers[model_type] = joblib.load(scaler_path)
            print(f"'{model_type}' ({timeframe}) 모델 로드 완료.")

    def predict(self, X_raw: np.ndarray, strategy: str = 'hard_voting') -> np.ndarray:
        all_predictions = []
        
        for model_type in self.model_list:
            scaler = self.scalers[model_type]
            session = self.models[model_type]
            
            B, T, F = X_raw.shape
            X_scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(B, T, F).astype(np.float32)
            
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: X_scaled}
            ort_outs = session.run(None, ort_inputs)
            
            if strategy == 'soft_voting':
                all_predictions.append(self._softmax(ort_outs[0]))
            else:
                all_predictions.append(np.argmax(ort_outs[0], axis=1))

        if strategy == 'soft_voting':
            avg_probs = np.mean(np.array(all_predictions), axis=0)
            final_preds = np.argmax(avg_probs, axis=1)
        else:
            predictions_stack = np.stack(all_predictions, axis=1)
            final_preds = np.array([np.bincount(row).argmax() for row in predictions_stack])
            
        return final_preds

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
