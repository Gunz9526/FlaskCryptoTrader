import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from feature_engineering import enhance_sequences_with_features
from data_processor import PriceSeqDataset

def prepare_dataloaders(
    df_main: pd.DataFrame, 
    higher_dfs: dict,
    seq_len: int, 
    batch_size: int,
    use_sampler: bool,
    labeling_params: dict,
    regime_series: pd.Series | None = None,
    regime_filter: str | None = None,
) -> dict:
    logging.info("--- Starting Data Preparation (Labeling: Triple Barrier) ---")
    X, y, timestamps = enhance_sequences_with_features(
        df_main=df_main,
        higher_dfs=higher_dfs,
        seq_len=seq_len,
        labeling_params=labeling_params
    )
    if len(X) == 0:
        logging.error("No sequences were generated. Aborting.")
        return None

    if regime_series is not None and regime_filter is not None:
        rs = regime_series
        try:
            reg_labels = rs.reindex(pd.Index(timestamps, name=rs.index.name))
        except Exception:
            reg_labels = rs.reindex(pd.Index(timestamps))
        mask = (reg_labels.values == regime_filter)
        kept = int(mask.sum())
        logging.info(f"Regime filter '{regime_filter}': keep {kept}/{len(X)} sequences")
        if kept < 10:
            logging.warning("Too few samples after regime filtering. Aborting.")
            return None
        X, y, timestamps = X[mask], y[mask], timestamps[mask]

    n = len(X)
    val_ratio = 0.1
    test_ratio = 0.1
    test_cut = int(n * (1 - test_ratio))
    val_cut = int(test_cut * (1 - val_ratio))

    X_train, y_train = X[:val_cut], y[:val_cut]
    X_val, y_val = X[val_cut:test_cut], y[val_cut:test_cut]
    X_test, y_test = X[test_cut:], y[test_cut:]

    scaler = StandardScaler().fit(X_train.reshape(-1, X.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X.shape[-1])).reshape(X_test.shape)

    train_ds = PriceSeqDataset(X_train, y_train)
    val_ds = PriceSeqDataset(X_val, y_val)
    test_ds = PriceSeqDataset(X_test, y_test)

    sampler = None
    if use_sampler:
        counts = np.bincount(y_train, minlength=3)
        weights = 1.0 / np.where(counts == 0, 1, counts)
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return {
        "train_loader": train_dl,
        "val_loader": val_dl,
        "test_loader": test_dl,
        "scaler": scaler,
        "input_size": X.shape[-1],
    }

def prepare_walkforward_folds(
    df_main: pd.DataFrame,
    higher_dfs: dict,
    seq_len: int,
    batch_size: int,
    use_sampler: bool,
    labeling_params: dict,
    n_folds: int = 3,
    val_ratio: float = 0.1,
    min_train_ratio: float = 0.6,
    regime_series: pd.Series | None = None,
    regime_filter: str | None = None,
) -> list[dict]:
    logging.info(f"--- Building Walk-Forward folds (n_folds={n_folds}, val_ratio={val_ratio}, min_train_ratio={min_train_ratio}) ---")
    X, y, timestamps = enhance_sequences_with_features(
        df_main=df_main,
        higher_dfs=higher_dfs,
        seq_len=seq_len,
        labeling_params=labeling_params
    )
    if len(X) == 0:
        logging.error("No sequences were generated. Aborting walk-forward preparation.")
        return []

    if regime_series is not None and regime_filter is not None:
        rs = regime_series
        try:
            reg_labels = rs.reindex(pd.Index(timestamps, name=rs.index.name))
        except Exception:
            reg_labels = rs.reindex(pd.Index(timestamps))
        mask = (reg_labels.values == regime_filter)
        kept = int(mask.sum())
        logging.info(f"[WF] Regime filter '{regime_filter}': keep {kept}/{len(X)} sequences")
        if kept < 10:
            logging.warning("Too few samples after regime filtering. Aborting WF.")
            return []
        X, y, timestamps = X[mask], y[mask], timestamps[mask]

    n = len(X)
    val_len = max(1, int(n * val_ratio))
    start_train_end = max(val_len, int(n * min_train_ratio))
    folds = []
    for k in range(n_folds):
        train_end = start_train_end + k * val_len
        val_start = train_end
        val_end = min(val_start + val_len, n)
        if val_end - val_start < 1 or train_end < 1:
            break

        X_train_raw = X[:train_end]
        y_train = y[:train_end]
        X_val_raw = X[val_start:val_end]
        y_val = y[val_start:val_end]

        scaler = StandardScaler().fit(X_train_raw.reshape(-1, X.shape[-1]))
        X_train = scaler.transform(X_train_raw.reshape(-1, X.shape[-1])).reshape(X_train_raw.shape)
        X_val = scaler.transform(X_val_raw.reshape(-1, X.shape[-1])).reshape(X_val_raw.shape)

        train_ds = PriceSeqDataset(X_train, y_train)
        val_ds = PriceSeqDataset(X_val, y_val)

        sampler = None
        if use_sampler:
            counts = np.bincount(y_train, minlength=3)
            weights = 1.0 / np.where(counts == 0, 1, counts)
            sample_weights = weights[y_train]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None), drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        folds.append({
            "train_loader": train_dl,
            "val_loader": val_dl,
            "scaler": scaler,
            "input_size": X.shape[-1],
            "fold_idx": k,
        })
    return folds