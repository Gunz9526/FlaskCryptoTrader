from app.redis_client import redis_client
from datetime import datetime
import pytz

def mta_filter(current_price: float, ema_200_1h: float, signal: int) -> bool:
    if signal == 1 and current_price > ema_200_1h:
        return True
    if signal == -1 and current_price < ema_200_1h:
        return True
    return False

def correlation_filter(new_asset_symbol: str, portfolio: dict) -> bool:
    if not portfolio:
        return True
    return True

def system_status_filter(symbol: str, current_positions: int) -> bool:
    if redis_client.get("circuit_breaker_active"):
        return False
    
    if redis_client.get("consecutive_loss_limit_active"):
        return False

    if redis_client.get(f"cooldown:{symbol}"):
        return False

    MAX_CONCURRENT_POSITIONS = 3
    if current_positions >= MAX_CONCURRENT_POSITIONS:
        return False
        
    return True

def volatility_filter(current_atr: float, avg_atr: float, multiplier: float = 2.5) -> bool:
    return current_atr <= avg_atr * multiplier

def momentum_filter(signal: int, momentum: float, roc: float) -> bool:
    if signal == 1 and momentum > 0 and roc > 0.5:
        return True
    if signal == -1 and momentum < 0 and roc < -0.5:
        return True
    return False

def volume_filter(volume_ratio: float, min_ratio: float = 1.2) -> bool:
    return volume_ratio >= min_ratio

def trend_confirmation_filter(signal: int, market_structure: int, adx: float) -> bool:
    if signal == 1 and market_structure == 1 and adx > 20:
        return True
    if signal == -1 and market_structure == -1 and adx > 20:
        return True
    return False