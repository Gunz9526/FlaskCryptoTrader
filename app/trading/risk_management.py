from app.redis_client import redis_client
import logging
from app.config import settings

def check_and_update_consecutive_losses(trade_is_loss: bool):
    key = "consecutive_losses"
    if trade_is_loss:
        losses = redis_client.incr(key)
        if losses >= settings.CONSECUTIVE_LOSS_LIMIT:
            logging.warning(f"{settings.CONSECUTIVE_LOSS_LIMIT} consecutive losses detected. Halting trading for {settings.TRADING_HALT_DURATION_SECONDS} seconds.")
            redis_client.set("consecutive_loss_limit_active", "true", ex=settings.TRADING_HALT_DURATION_SECONDS)
    else:
        redis_client.delete(key)

def set_post_trade_cooldown(symbol: str):
    redis_client.set(f"cooldown:{symbol}", "true", ex=1800)

def calculate_stop_loss(entry_price: float, atr: float, side: str, model_type: str) -> float:
    multiplier = 2.5 if model_type == 'trending' else 1.8
    
    if side == 'long':
        return entry_price - (multiplier * atr)
    elif side == 'short':
        return entry_price + (multiplier * atr)
    raise ValueError("side는 'long' 또는 'short'여야 합니다.")

def calculate_position_size(
    account_equity: float, risk_per_trade_ratio: float,
    entry_price: float, stop_loss_price: float
) -> float:
    risk_amount_per_trade = account_equity * risk_per_trade_ratio
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        logging.warning("Risk per unit is zero. Cannot calculate position size.")
        return 0.0
    position_size = risk_amount_per_trade / risk_per_unit
    return round(position_size, 8)