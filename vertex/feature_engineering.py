import talib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import logging


def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    epsilon = 1e-10 

    ema_20 = talib.EMA(close, timeperiod=20)
    ema_50 = talib.EMA(close, timeperiod=50)
    sma_200 = talib.SMA(close, timeperiod=200)
    df['price_vs_sma200'] = (close / (sma_200 + epsilon)) - 1
    df['ema_20_vs_50'] = (ema_20 / (ema_50 + epsilon)) - 1
    df['market_structure'] = np.where((close > ema_20) & (ema_20 > ema_50), 1,
                                    np.where((close < ema_20) & (ema_20 < ema_50), -1, 0))
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)

    df['rsi'] = talib.RSI(close, timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    
    stoch_k, stoch_d = talib.STOCH(high, low, close)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + epsilon)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + epsilon)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    volume_sma_20 = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / (volume_sma_20 + epsilon)

    returns = np.log(close / close.shift(1))
    df['ret_1'] = returns
    df['ret_vol_20'] = returns.rolling(20).std()
    df['roc_10'] = talib.ROC(close, timeperiod=10)
    try:
        df['kama_10'] = talib.KAMA(close, timeperiod=10)
        df['price_vs_kama10'] = (close / (df['kama_10'] + epsilon)) - 1
    except Exception:
        df['kama_10'] = 0.0
        df['price_vs_kama10'] = 0.0
    df['atr_change_10'] = df['atr'].pct_change(10)
    df['atr_pct_100'] = df['atr'].rolling(100).apply(
        lambda s: s.rank(pct=True).iloc[-1] if s.notna().all() else np.nan, raw=False
    )
    vol_med_40 = volume.rolling(40).median()
    df['volume_spike'] = volume / (vol_med_40 + epsilon)
    df['vol_pct_100'] = volume.rolling(100).apply(
        lambda s: s.rank(pct=True).iloc[-1] if s.notna().all() else np.nan, raw=False
    )
    df['rsi_regime'] = np.where(df['rsi'] > 60, 1, np.where(df['rsi'] < 40, -1, 0))
    df['bb_squeeze_100'] = df['bb_width'].rolling(100).apply(
        lambda s: s.rank(pct=True).iloc[-1] if s.notna().all() else np.nan, raw=False
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    returns = np.log(close / close.shift(1))
    return returns.ewm(span=span).std()


def _apply_triple_barrier(
    close: pd.Series, 
    events: pd.DataFrame, 
    pt_sl: List[float], 
    molecule: pd.Series
) -> pd.DataFrame:
    out = events[['t1']].copy(deep=True)
    
    out['sl_time'] = pd.NaT
    out['pt_time'] = pd.NaT
    
    pt = pt_sl[0] * molecule if pt_sl[0] > 0 else pd.Series(index=events.index)
    sl = -pt_sl[1] * molecule if pt_sl[1] > 0 else pd.Series(index=events.index)

    for loc, t1 in events['t1'].items():
        if pd.isna(t1): continue
        
        path_prices = close[loc:t1]
        returns = (path_prices / close[loc] - 1)
        
        sl_hits = returns[returns < sl[loc]]
        if not sl_hits.empty:
            out.loc[loc, 'sl_time'] = sl_hits.index.min()

        pt_hits = returns[returns > pt[loc]]
        if not pt_hits.empty:
            out.loc[loc, 'pt_time'] = pt_hits.index.min()
    
    return out


def enhance_sequences_with_features(
    df_main: pd.DataFrame,
    higher_dfs: Dict[str, pd.DataFrame],
    seq_len: int,
    labeling_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info(f"--- Starting Feature & Sequence Generation (Labeling: triple_barrier) ---")
    
    df_main_featured = _calculate_technical_indicators(df_main).ffill()
    df_merged = df_main_featured.copy()
    
    for tf, df_h in higher_dfs.items():
        if df_h is None or df_h.empty:
            logging.warning(f"Higher timeframe dataframe for '{tf}' is empty. Skipping.")
            continue
        
        logging.info(f"Calculating and merging features from '{tf}' timeframe...")
        df_h_featured = _calculate_technical_indicators(df_h).add_suffix(f'_{tf}').ffill()
        df_merged = pd.merge_asof(
            left=df_merged, right=df_h_featured,
            left_index=True, right_index=True, direction='backward'
        )
    df_merged = df_merged.ffill().fillna(0)

    epsilon = 1e-10
    for tf in higher_dfs.keys():
        if f'rsi_{tf}' in df_merged.columns:
            df_merged[f'rsi_divergence_{tf}'] = df_merged['rsi'] - df_merged[f'rsi_{tf}']

        if f'atr_{tf}' in df_merged.columns and 'atr' in df_merged.columns:
            df_merged[f'atr_ratio_{tf}'] = df_merged['atr'] / (df_merged[f'atr_{tf}'] + epsilon)


    base_features = [
        'price_vs_sma200', 'ema_20_vs_50', 'market_structure', 'adx',
        'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'macd', 'macdsignal', 'macdhist',
        'bb_width', 'bb_position', 'atr', 'volume_ratio',        
        'ret_1', 'ret_vol_20', 'roc_10', 'kama_10', 'price_vs_kama10',
        'atr_change_10', 'atr_pct_100', 'volume_spike', 'vol_pct_100',
        'rsi_regime', 'bb_squeeze_100'
    ]
    final_feature_columns = base_features.copy()
    for tf in higher_dfs.keys():
        final_feature_columns.extend([
            f'price_vs_sma200_{tf}', f'adx_{tf}', f'rsi_{tf}',
            f'atr_ratio_{tf}'
        ])
    
    for tf in higher_dfs.keys():
        if f'atr_{tf}' in df_merged.columns:
            df_merged[f'atr_ratio_{tf}'] = df_merged['atr'] / (df_merged[f'atr_{tf}'] + epsilon)
    
    final_feature_columns = sorted([col for col in final_feature_columns if col in df_merged.columns])
    logging.info(f"Using {len(final_feature_columns)} refined features for DL models.")
    
    X_raw = df_merged[final_feature_columns].to_numpy(dtype=np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    labels = pd.Series(0, index=df_merged.index, dtype=np.int64)
    
    pt_sl = labeling_params.get('pt_sl', [1.5, 1.0])
    look_forward_candles = labeling_params.get('look_forward_candles', 12)
    volatility_span = labeling_params.get('volatility_span', 100)

    close = df_merged['close']
    volatility = _get_daily_volatility(close, span=volatility_span)
    
    events = pd.DataFrame(index=df_merged.index)
    t1_indices = np.minimum(np.arange(len(df_merged)) + look_forward_candles, len(df_merged) - 1)
    events['t1'] = df_merged.index[t1_indices]
    events = events[events.index < events.t1]

    barriers = _apply_triple_barrier(close, events, pt_sl, volatility)
    
    barriers[['pt_time', 'sl_time']] = barriers[['pt_time', 'sl_time']].apply(pd.to_datetime)
    
    first_touch_times = barriers[['pt_time', 'sl_time']].min(axis=1)
    
    pt_win_mask = (barriers['pt_time'] == first_touch_times) & barriers['pt_time'].notna()
    sl_win_mask = (barriers['sl_time'] == first_touch_times) & barriers['sl_time'].notna()
    
    pt_win_mask = pt_win_mask.reindex(labels.index, fill_value=False)
    sl_win_mask = sl_win_mask.reindex(labels.index, fill_value=False)

    labels.loc[pt_win_mask] = 1
    labels.loc[sl_win_mask] = 2


    limit = len(X_raw) - seq_len
    X_seq, y_seq, timestamps = [], [], []

    y_raw = labels.to_numpy()

    for i in range(limit):
        label_index = i + seq_len - 1
        X_seq.append(X_raw[i:i+seq_len])
        y_seq.append(y_raw[label_index])
        timestamps.append(df_merged.index[label_index])

    if not X_seq:
        logging.error("No sequences were generated. Check data length and parameters.")
        return np.array([]), np.array([]), np.array([])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.int64)
    timestamps = np.asarray(timestamps)
    
    unique, counts = np.unique(y_seq, return_counts=True)
    class_dist = {f"Class {k}": v for k, v in zip(unique, counts)}
    logging.info(f"Generated {len(X_seq)} sequences. Class distribution: {class_dist}")
    
    return X_seq, y_seq, timestamps