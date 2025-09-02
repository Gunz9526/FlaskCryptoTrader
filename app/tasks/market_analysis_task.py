import pandas as pd
import talib
import logging
from .celery_app import celery_instance
from app.metrics import MARKET_REGIME_GAUGE
from app.models import MarketData
from app.redis_client import redis_client
from app.extensions import db
from sqlalchemy import text


@celery_instance.task(name="tasks.update_market_regime")
def update_market_regime(symbol: str):
    logging.info(f"Updating market regime for {symbol}...")
    try:
        stmt_1h = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '1h' ORDER BY timestamp DESC LIMIT 200")
        stmt_15m = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '15m' ORDER BY timestamp DESC LIMIT 200")
        
        df_1h = pd.read_sql(stmt_1h, db.engine, params={'symbol': symbol}, index_col='timestamp').sort_index()
        df_15m = pd.read_sql(stmt_15m, db.engine, params={'symbol': symbol}, index_col='timestamp').sort_index()

        if len(df_1h) < 50 or len(df_15m) < 50:
            logging.warning("Not enough data to determine market regime.")
            return
        
        df_1h['adx'] = talib.ADX(df_1h['high'], df_1h['low'], df_1h['close'], timeperiod=14)
        df_1h['atr_norm'] = talib.ATR(df_1h['high'], df_1h['low'], df_1h['close'], timeperiod=14) / df_1h['close']

        upper, middle, lower = talib.BBANDS(df_15m['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df_15m['bb_width'] = (upper - lower) / middle
        
        df_1h.dropna(inplace=True)
        df_15m.dropna(inplace=True)

        if df_1h.empty or df_15m.empty:
            logging.warning("Not enough data after indicator calculation.")
            return

        latest_1h = df_1h.iloc[-1]
        latest_15m = df_15m.iloc[-1]
        
        regime = "ranging"
        
        if latest_1h['atr_norm'] > df_1h['atr_norm'].rolling(50).mean().iloc[-1] * 2.5:
            regime = "chaotic_expansion"
        
        elif latest_15m['bb_width'] < df_15m['bb_width'].quantile(0.1):
            regime = "ranging_squeeze"

        elif latest_1h['adx'] > 28:
            regime = "trending_strong"
        elif latest_1h['adx'] > 20:
            regime = "trending_weak"
            
        else:
            if latest_1h['atr_norm'] > df_1h['atr_norm'].quantile(0.6):
                regime = "ranging_volatile"
            else:
                regime = "ranging_quiet"

        redis_client.set(f"market_regime:{symbol}", regime)
        
        regime_map = {
            "trending_strong": 5, "trending_weak": 4,
            "ranging_volatile": 3, "ranging_quiet": 2, "ranging_squeeze": 1,
            "chaotic_expansion": 0
        }
        MARKET_REGIME_GAUGE.labels(symbol=symbol).set(regime_map.get(regime, 2))
        
        logging.info(f"Market regime for {symbol} updated to: {regime} (ADX: {latest_1h['adx']:.2f}, ATR_Norm: {latest_1h['atr_norm']:.4f}, BBW: {latest_15m['bb_width']:.4f})")

    except Exception as e:
        logging.error(f"Error updating market regime: {e}", exc_info=True)