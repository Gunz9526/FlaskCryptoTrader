import pandas as pd
import talib
import numpy as np

def add_technical_indicators(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    df_1h = df_1h.copy()
    df_1h['ema_200_1h'] = talib.EMA(df_1h['close'], timeperiod=200)
    df_1h['rsi_1h'] = talib.RSI(df_1h['close'], timeperiod=14)
    df_1h['adx_1h'] = talib.ADX(df_1h['high'], df_1h['low'], df_1h['close'], timeperiod=14)
    df_1h['atr_1h'] = talib.ATR(df_1h['high'], df_1h['low'], df_1h['close'], timeperiod=14)
    
    df_15m_enhanced = df_15m.copy()
    
    df_15m_enhanced['ema_20'] = talib.EMA(df_15m['close'], timeperiod=20)
    df_15m_enhanced['ema_50'] = talib.EMA(df_15m['close'], timeperiod=50)
    df_15m_enhanced['sma_200'] = talib.SMA(df_15m['close'], timeperiod=200)
    
    df_15m_enhanced['rsi'] = talib.RSI(df_15m['close'], timeperiod=14)
    df_15m_enhanced['macd'], df_15m_enhanced['macdsignal'], df_15m_enhanced['macdhist'] = talib.MACD(
        df_15m['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df_15m_enhanced['stoch_k'], df_15m_enhanced['stoch_d'] = talib.STOCH(
        df_15m['high'], df_15m['low'], df_15m['close']
    )
    df_15m_enhanced['williams_r'] = talib.WILLR(
        df_15m['high'], df_15m['low'], df_15m['close'], timeperiod=14
    )
    
    df_15m_enhanced['adx'] = talib.ADX(df_15m['high'], df_15m['low'], df_15m['close'], timeperiod=14)
    df_15m_enhanced['atr'] = talib.ATR(df_15m['high'], df_15m['low'], df_15m['close'], timeperiod=14)
    df_15m_enhanced['bb_upper'], df_15m_enhanced['bb_middle'], df_15m_enhanced['bb_lower'] = talib.BBANDS(
        df_15m['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df_15m_enhanced['cci'] = talib.CCI(df_15m['high'], df_15m['low'], df_15m['close'], timeperiod=14)
    
    df_15m_enhanced['bb_position'] = (df_15m['close'] - df_15m_enhanced['bb_lower']) / (
        df_15m_enhanced['bb_upper'] - df_15m_enhanced['bb_lower']
    )
    df_15m_enhanced['bb_width'] = (df_15m_enhanced['bb_upper'] - df_15m_enhanced['bb_lower']) / df_15m_enhanced['bb_middle']
    
    df_15m_enhanced['volume_sma'] = talib.SMA(df_15m['volume'].astype(float), timeperiod=20)
    df_15m_enhanced['volume_ratio'] = df_15m['volume'] / df_15m_enhanced['volume_sma']
    df_15m_enhanced['obv'] = talib.OBV(df_15m['close'], df_15m['volume'].astype(float))
    df_15m_enhanced['obv_sma'] = talib.SMA(df_15m_enhanced['obv'], timeperiod=20)
    df_15m_enhanced['obv_ratio'] = df_15m_enhanced['obv'] / df_15m_enhanced['obv_sma']
    
    df_15m_enhanced['price_change'] = df_15m['close'].pct_change()
    df_15m_enhanced['volatility'] = df_15m_enhanced['price_change'].rolling(window=20).std()
    df_15m_enhanced['momentum'] = talib.MOM(df_15m['close'], timeperiod=10)
    df_15m_enhanced['roc'] = talib.ROC(df_15m['close'], timeperiod=10)
    
    df_final = pd.merge_asof(
        df_15m_enhanced.sort_index(),
        df_1h[['ema_200_1h', 'rsi_1h', 'adx_1h', 'atr_1h']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    df_final['price_vs_ema200_1h'] = df_final['close'] / df_final['ema_200_1h'] - 1
    df_final['ema_20_vs_50'] = df_final['ema_20'] / df_final['ema_50'] - 1
    df_final['rsi_divergence'] = df_final['rsi'] - df_final['rsi_1h']
    df_final['atr_ratio'] = df_final['atr'] / df_final['atr_1h']
    df_final['adx_trend_strength'] = np.where(df_final['adx'] > 25, 1, 0)
    df_final['market_structure'] = np.where(
        (df_final['close'] > df_final['ema_20']) & (df_final['ema_20'] > df_final['ema_50']), 1,
        np.where((df_final['close'] < df_final['ema_20']) & (df_final['ema_20'] < df_final['ema_50']), -1, 0)
    )
    
    return df_final

def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        'rsi', 'macd', 'macdsignal', 'macdhist', 'stoch_k', 'stoch_d', 'williams_r',
        'adx', 'atr', 'cci', 'bb_position', 'bb_width', 'volume_ratio', 'obv_ratio',
        'volatility', 'momentum', 'roc', 'price_vs_ema200_1h', 'ema_20_vs_50', 
        'rsi_divergence', 'atr_ratio', 'adx_trend_strength', 'market_structure',
        'rsi_1h', 'adx_1h'
    ]
    
    available_columns = [col for col in feature_columns if col in df.columns]
    return df[available_columns].dropna()

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
    df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
    df['macd_signal'] = np.where(df['macd'] > df['macdsignal'], 1, -1)
    df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8, 1, 0)
    df['volume_spike'] = np.where(df['volume_ratio'] > 2.0, 1, 0)
    
    df['trend_alignment'] = np.where(
        (df['market_structure'] == 1) & (df['adx'] > 25) & (df['price_vs_ema200_1h'] > 0), 1,
        np.where((df['market_structure'] == -1) & (df['adx'] > 25) & (df['price_vs_ema200_1h'] < 0), -1, 0)
    )
    
    df['reversal_signal'] = np.where(
        (df['rsi'] < 30) & (df['williams_r'] < -80) & (df['bb_position'] < 0.2), 1,
        np.where((df['rsi'] > 70) & (df['williams_r'] > -20) & (df['bb_position'] > 0.8), -1, 0)
    )
    
    return df



def get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    returns = np.log(close / close.shift(1))
    return returns.ewm(span=span).std()

def apply_triple_barrier(close: pd.Series, events: pd.DataFrame, pt_sl: list, molecule: pd.Series) -> pd.DataFrame:
    out = events[['t1']].copy(deep=True)
    
    if pt_sl[0] > 0:
        pt = pt_sl[0] * molecule
    else:
        pt = pd.Series(index=events.index)
    
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * molecule
    else:
        sl = pd.Series(index=events.index)

    for loc, t1 in events['t1'].items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1)
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    
    return out

def create_labels(df: pd.DataFrame, regime: str) -> pd.Series:
    close = df['close']
    vol = get_daily_volatility(close)
    
    events = pd.DataFrame(index=df.index)
    
    avg_atr_as_pct = (df['atr'] / df['close']).rolling(window=100, min_periods=20).mean()
    target_move_pct = 0.015
    
    look_forward_period = (target_move_pct / avg_atr_as_pct).round().fillna(12)
    look_forward_period = look_forward_period.clip(lower=4, upper=24)
    
    t1_series = df.index.to_series().apply(lambda x: df.index[df.index.get_loc(x) + int(look_forward_period.get(x, 12))] if df.index.get_loc(x) + int(look_forward_period.get(x, 12)) < len(df.index) else None)
    events['t1'] = t1_series
    
    events = events.dropna(subset=['t1'])

    pt_sl_ratio = [2, 1] if regime == 'trending' else [1, 1]

    
    daily_vol = vol.reindex(events.index, method='ffill')
    barriers = apply_triple_barrier(close, events, pt_sl_ratio, daily_vol)

    barriers['out'] = 0
    pt_win = barriers.pt.notna() & barriers.sl.notna() & (barriers.pt < barriers.sl) | barriers.pt.notna() & barriers.sl.isna()
    sl_win = barriers.pt.notna() & barriers.sl.notna() & (barriers.sl < barriers.pt) | barriers.sl.notna() & barriers.pt.isna()
    
    barriers.loc[pt_win, 'out'] = 1
    barriers.loc[sl_win, 'out'] = -1
    
    return barriers['out']
