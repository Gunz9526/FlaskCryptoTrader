import torch
import torch.nn as nn
import numpy as np
import logging
from copy import deepcopy
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from data_processor import PriceSeqDataset
from evaluate_model import find_best_thresholds

class EarlyStopping:
    def __init__(self, monitor: str = "val_f1", mode: str = "max", patience: int = 5, min_delta: float = 1e-4):
        self.monitor, self.mode, self.patience, self.min_delta = monitor, mode, patience, min_delta
        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad_epochs = 0
        self.best_state_dict = None

    def step(self, metrics: dict, model: nn.Module) -> bool:
        val = metrics[self.monitor]
        improved = (val > self.best + self.min_delta) if self.mode == "max" else (val < self.best - self.min_delta)
        if improved:
            self.best = val
            self.num_bad_epochs = 0
            self.best_state_dict = deepcopy(model.state_dict())
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

def _eval_epoch(model: nn.Module, dl: DataLoader, device: str, threshold_grid: list[float] | None = None) -> dict:
    model.eval()
    y_true, y_pred, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            y_pred.extend(probs.argmax(axis=1))
            y_true.extend(yb.numpy())
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3), dtype=np.float32)

    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1_weighted,
        "val_f1_macro": f1_macro
    }

    if threshold_grid is not None and len(all_probs) > 0:
        best_th, best_score = find_best_thresholds(y_true, all_probs, grid=threshold_grid, metric="macro_f1")
        metrics["val_f1_macro_thr"] = best_score
        metrics["best_thresholds"] = best_th
    return metrics

def create_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model: nn.Module, X_train, y_train, X_val, y_val, epochs, batch_size, lr, patience, device, 
                model_type: str = None, use_weighted_sampler: bool = False, weight_decay: float = 1e-4,
                scheduler_params: dict | None = None, label_smoothing: float = 0.05,
                early_metric: str = "val_f1_macro_thr") -> tuple[nn.Module, float]:
    logging.info(f"Training model: {model.__class__.__name__} on {device}")
    
    counts = np.bincount(y_train, minlength=3)
    weights = len(y_train) / (3 * np.where(counts == 0, 1, counts))
    
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device),
        label_smoothing=label_smoothing
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_weighted_sampler:
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)       
        logging.info("Using WeightedRandomSampler for class imbalance")
        train_dl = DataLoader(PriceSeqDataset(X_train, y_train), batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        train_dl = DataLoader(PriceSeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)   
    val_dl = DataLoader(PriceSeqDataset(X_val, y_val), batch_size=batch_size)

    if model_type in ("transformer", "patchtst"):
        total_steps = max(1, len(train_dl) * epochs)
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = create_warmup_scheduler(opt, warmup_steps, total_steps)
        logging.info(f"Using step-wise warmup scheduler for {model_type} (warmup_steps: {warmup_steps}, total_steps: {total_steps})")
    else:
        if scheduler_params is None:
            scheduler_params = {"T_0": 10, "T_mult": 1}
            logging.warning(f"Scheduler params not provided. Using default values: {scheduler_params}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, **scheduler_params)
        logging.info(f"Using CosineAnnealingWarmRestarts scheduler with params: {scheduler_params}")

    stopper = EarlyStopping(monitor=early_metric, patience=patience)
    scaler = GradScaler(enabled=(device == "cuda"))

    threshold_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] if early_metric == "val_f1_macro_thr" else None
    
    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device, enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            if model_type in ("transformer", "patchtst"):
                scheduler.step()
            batch_losses.append(loss.item())

        if model_type not in ("transformer", "patchtst"):
            scheduler.step()
        
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        metrics = _eval_epoch(model, val_dl, device, threshold_grid=threshold_grid)

        if "val_f1_macro_thr" in metrics:
            logging.info(f"[Epoch {ep:02d}] loss={epoch_loss:.4f}, f1_macro_thr={metrics['val_f1_macro_thr']:.4f}, "
                         f"f1_macro={metrics['val_f1_macro']:.4f}, lr={opt.param_groups[0]['lr']:.2e}")
        else:
            logging.info(f"[Epoch {ep:02d}] loss={epoch_loss:.4f}, f1_macro={metrics['val_f1_macro']:.4f}, "
                         f"f1_w={metrics['val_f1']:.4f}, lr={opt.param_groups[0]['lr']:.2e}")
        
        if stopper.step(metrics, model):
            logging.info(f"Early stopping triggered. Best {stopper.monitor}: {stopper.best:.4f}")
            break
    
    if stopper.best_state_dict:
        model.load_state_dict(stopper.best_state_dict)
    return model, stopper.best