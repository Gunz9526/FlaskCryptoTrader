from app.redis_client import redis_client
import logging

def check_and_update_consecutive_losses(trade_is_loss: bool):
    """
    거래 종료 후 연속 손실을 확인, 카운터 업데이트
    5회 연속 손실 시 24시간 거래 중단
    """
    key = "consecutive_losses"
    if trade_is_loss:
        losses = redis_client.incr(key)
        if losses >= 5:
            logging.warning("5 consecutive losses detected. Halting trading for 24 hours.")
            redis_client.set("consecutive_loss_limit_active", "true", ex=86400)
        redis_client.delete(key)

def set_post_trade_cooldown(symbol: str):
    """거래 후 30분 쿨타임"""
    redis_client.set(f"cooldown:{symbol}", "true", ex=1800)

def calculate_stop_loss(entry_price: float, atr: float, side: str) -> float:
    if side == 'long':
        return entry_price - (2 * atr)
    elif side == 'short':
        return entry_price + (2 * atr)
    raise ValueError("side는 'long' 또는 'short'여야 합니다.")

def calculate_position_size(
    account_equity: float, risk_per_trade_ratio: float,
    entry_price: float, stop_loss_price: float
) -> float:
    risk_amount_per_trade = account_equity * risk_per_trade_ratio
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        print("Warning: Risk per unit is zero. Cannot calculate position size.")
        return 0.0

    position_size = risk_amount_per_trade / risk_per_unit
    return round(position_size, 8)