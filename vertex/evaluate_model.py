import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


def predict_proba(model: torch.nn.Module, dl: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            y_pred_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.extend(yb.numpy())
    return np.asarray(y_true), np.asarray(y_pred_probs, dtype=np.float32)

def _apply_thresholds(probs: np.ndarray, thresholds: dict | None) -> np.ndarray:
    if not thresholds:
        return np.argmax(probs, axis=1)
    th_buy = float(thresholds.get("buy", 0.5))
    th_sell = float(thresholds.get("sell", 0.5))
    th_hold = float(thresholds.get("hold", 0.0))
    p_hold, p_buy, p_sell = probs[:, 0], probs[:, 1], probs[:, 2]

    preds = np.zeros(len(probs), dtype=np.int64)
    buy_mask = p_buy >= th_buy
    sell_mask = p_sell >= th_sell
    hold_mask = p_hold >= th_hold

    both_mask = buy_mask & sell_mask
    buy_only = buy_mask & ~both_mask
    sell_only = sell_mask & ~both_mask

    preds[buy_only] = 1
    preds[sell_only] = 2
    preds[both_mask] = np.stack([p_hold[both_mask], p_buy[both_mask], p_sell[both_mask]], axis=1).argmax(axis=1)
    preds[~(buy_mask | sell_mask) & hold_mask] = 0
    return preds

def find_best_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: list[float] | None = None, metric: str = "macro_f1") -> tuple[dict, float]:
    if grid is None:
        grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_score, best = -1.0, {"buy": 0.5, "sell": 0.5}
    for tb in grid:
        for ts in grid:
            y_pred = _apply_thresholds(probs, {"buy": tb, "sell": ts})
            if metric == "macro_f1":
                score = f1_score(y_true, y_pred, average="macro", zero_division=0)
            else:
                score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            if score > best_score:
                best_score, best = score, {"buy": tb, "sell": ts}
    return best, best_score

def evaluate_model_comprehensive(model: torch.nn.Module, dl: DataLoader, device: str, thresholds: dict | None = None) -> dict:
    model.eval()
    y_true, y_pred_probs = predict_proba(model, dl, device)
    y_pred = _apply_thresholds(y_pred_probs, thresholds)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    class_metrics = {
        'Hold': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': support[0]},
        'Buy':  {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': support[1]},
        'Sell': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2], 'support': support[2]},
    }
    signal_accuracy = (precision[1] + precision[2]) / 2 if (support[1] + support[2]) > 0 else 0.0

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "class_metrics": class_metrics,
        "signal_accuracy": signal_accuracy
    }

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')