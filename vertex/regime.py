import numpy as np
import pandas as pd
import logging

try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

def _safe_pct_rank(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=max(5, win // 5)).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1] if w.notna().sum() > 0 else np.nan,
        raw=False
    )

def get_regime_series(df: pd.DataFrame, adx_thr: float = 22.0, squeeze_win: int = 100, mom_win: int = 10) -> pd.Series:
    df = df.copy()
    if not {'high','low','close'}.issubset(df.columns):
        raise ValueError("DataFrame must include ['high','low','close'] columns for regime detection.")
    h, l, c = df['high'], df['low'], df['close']
    eps = 1e-10

    if _HAS_TALIB:
        adx = talib.ADX(h.values, l.values, c.values, timeperiod=14)
        adx = pd.Series(adx, index=df.index)
    else:
        tr = (h - l).abs()
        tr1 = (h - c.shift(1)).abs()
        tr2 = (l - c.shift(1)).abs()
        tr = pd.concat([tr, tr1, tr2], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean()
        up = (c - c.shift(1)).clip(lower=0)
        down = (c.shift(1) - c).clip(lower=0)
        plus = 100 * (up.rolling(14, min_periods=7).sum() / (atr + eps))
        minus = 100 * (down.rolling(14, min_periods=7).sum() / (atr + eps))
        dx = 100 * (plus - minus).abs() / ((plus + minus) + eps)
        adx = dx.rolling(14, min_periods=7).mean()

    if _HAS_TALIB:
        mb, bb_up, bb_lo = talib.BBANDS(c.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        bb_up = pd.Series(bb_up, index=df.index)
        bb_lo = pd.Series(bb_lo, index=df.index)
    else:
        ma = c.rolling(20, min_periods=10).mean()
        sd = c.rolling(20, min_periods=10).std()
        bb_up, bb_lo = ma + 2 * sd, ma - 2 * sd
    bb_width = (bb_up - bb_lo) / (c.abs() + eps)
    bb_squeeze_pct = _safe_pct_rank(bb_width, squeeze_win)  # 낮으면 squeeze

    ret = np.log(c / c.shift(1))
    mom = ret.rolling(mom_win, min_periods=max(3, mom_win//2)).sum().abs()

    trending = (adx >= adx_thr) & (bb_squeeze_pct >= 0.5) & (mom >= mom.median(skipna=True))
    regime = pd.Series(np.where(trending, "trending", "ranging"), index=df.index).astype("category")
    regime = regime.where(~(adx.isna() | bb_squeeze_pct.isna() | mom.isna()))
    regime = regime.ffill().bfill()
    logging.info("Computed regime series: trending ratio=%.3f", (regime == "trending").mean())
    return regime

def latest_regime(df: pd.DataFrame) -> str:
    s = get_regime_series(df)
    return str(s.iloc[-1])