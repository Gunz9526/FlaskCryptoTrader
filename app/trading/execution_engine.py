import pandas as pd
import logging
from alpaca.trading.client import TradingClient
from sqlalchemy import select, text
from app.config import settings
from app.models import MarketData
from app.extensions import db
from app.redis_client import redis_client
from app.metrics import AI_PREDICTION_COUNTER
from app.trading.feature_engineering import prepare_features_for_model, create_advanced_features
from app.trading import feature_engineering, filters, risk_management, order_management
from app.ml.model_handler import ModelHandler
from app.trading.alpaca_client import get_trading_client

def run_trading_cycle_for_symbol(symbol: str):
    logging.info(f"===== Running Trading Cycle for {symbol} =====")
    try:
        query_15m = text("""
            SELECT * FROM market_data 
            WHERE symbol = :symbol AND timeframe = '15m' 
            ORDER BY timestamp DESC 
            LIMIT 500
        """)
        query_1h = text("""
            SELECT * FROM market_data 
            WHERE symbol = :symbol AND timeframe = '1h' 
            ORDER BY timestamp DESC 
            LIMIT 500
        """)

        df_15m = pd.read_sql(query_15m, db.session.connection(), params={'symbol': symbol}, index_col='timestamp').sort_index()
        df_1h = pd.read_sql(query_1h, db.session.connection(), params={'symbol': symbol}, index_col='timestamp').sort_index()

        if df_15m.empty or df_1h.empty or len(df_15m) < 100 or len(df_1h) < 50:
            logging.warning("Not enough data in DB to run cycle. Please run data collection tasks first.")
            return        

        regime_bytes = redis_client.get(f"market_regime:{symbol}")
        raw_regime = regime_bytes.decode('utf-8') if regime_bytes else "ranging"
        logging.info(f"Current market regime for {symbol}: {raw_regime}")

        features_df = feature_engineering.add_technical_indicators(df_15m, df_1h)
        if features_df.empty:
            logging.warning("Feature engineering resulted in empty dataframe. Skipping.")
            return
        
        features_df = create_advanced_features(features_df)
        model_features = prepare_features_for_model(features_df)
        if model_features.empty:
            logging.warning("No valid features for model. Skipping.")
            return
        
        latest_data = features_df.iloc[-1:]
        latest_features = model_features.iloc[-1:]

        def to_base_model_type(regime: str) -> str:
            return 'trending' if 'trend' in regime else 'ranging'
        
        base_model_type = to_base_model_type(raw_regime)

        key_indicators = {
            "Price": latest_data['close'].iloc[0],
            "EMA_200_1H": latest_data['ema_200_1h'].iloc[0],
            "ATR_15M": latest_data['atr'].iloc[0],
            "ATR_1H": latest_data['atr_1h'].iloc[0],
            "ADX_15M": latest_data['adx'].iloc[0],
            "RSI_15M": latest_data['rsi'].iloc[0],
            "Sentiment": (redis_client.get("news_sentiment_score") or b'0.0').decode()
        }
        logging.info(f"Key Indicators Snapshot: {key_indicators}")

        if raw_regime == "chaotic_expansion":
            logging.warning("Market is in chaotic expansion. No new trades will be placed.")
            return
        
        model_handler = ModelHandler(symbol=symbol, model_type=base_model_type)
        signal = model_handler.get_ensemble_prediction(features=latest_features)
        confidence = model_handler.get_prediction_confidence(features=latest_features)
        
        signal_str = {1: 'buy', -1: 'sell', 0: 'neutral'}.get(signal, 'unknown')
        AI_PREDICTION_COUNTER.labels(symbol=symbol, regime=raw_regime, signal=signal_str).inc()
        
        if signal == 0:
            logging.info(f"Filter 1: AI signal is '{signal_str}'. No action taken.")
            return
        logging.info(f"Filter 1: AI signal is '{signal_str}' with confidence {confidence:.3f}.")

        if not filters.mta_filter(latest_data['close'].iloc[0], latest_data['ema_200_1h'].iloc[0], signal):
            logging.warning("Filter 2: MTA filter failed. Signal rejected.")
            return
        logging.info("Filter 2: MTA filter passed.")

        trading_client = get_trading_client()
        positions = trading_client.get_all_positions()
        if not filters.system_status_filter(symbol, len(positions)):
            return
        logging.info("Filter 3: Portfolio & System status filters passed.")

        atr_rolling_mean = features_df['atr'].rolling(20).mean().iloc[-1]
        current_atr = latest_data['atr'].iloc[0]
        if not filters.volatility_filter(current_atr, atr_rolling_mean):
            logging.warning("Filter 4: Volatility too high. Signal rejected.")
            return
        logging.info("Filter 4: Volatility filter passed.")

        logging.info("All filters passed. Proceeding to order execution.")
        side = 'long' if signal == 1 else 'short'
        account = trading_client.get_account()
                
        current_price = latest_data['close'].iloc[0]
        current_atr = latest_data['atr'].iloc[0]

        stop_loss_price = risk_management.calculate_stop_loss(current_price, current_atr, side, base_model_type)

        slippage_allowance = current_atr * 0.1 / current_price

        if side == 'long':
            limit_price = current_price * (1 + slippage_allowance)
        else:
            limit_price = current_price * (1 - slippage_allowance)

        position_size = risk_management.calculate_position_size(
            float(account.equity), 
            settings.RISK_PER_TRADE_RATIO,
            current_price, 
            stop_loss_price
        )

        if position_size <= 0:
            logging.warning("Calculated position size is zero or less. Cannot place trade.")
            return

        submitted_order = order_management.execute_bracket_order(symbol, position_size, side, stop_loss_price, limit_price)

        if submitted_order:
            pos_key = f"position:{symbol}"
            pos_data = {
                "entry_price": latest_data['close'].iloc[0],
                "stop_loss": stop_loss_price,
                "side": side,
                "qty": position_size,
                "confidence": confidence,
                "regime": raw_regime
            }
            redis_client.hset(pos_key, mapping=pos_data)
            redis_client.expire(pos_key, 86400)
            logging.info(f"Position data for {symbol} saved to Redis for dynamic management.")

    except Exception as e:
        logging.error(f"An error occurred in the trading cycle for {symbol}: {e}", exc_info=True)