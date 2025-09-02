from alpaca.trading.client import TradingClient
from app.config import settings

_trading_client = None

def get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(
            settings.ALPACA_API_KEY, 
            settings.ALPACA_SECRET_KEY, 
            paper=settings.ALPACA_PAPER
        )
    return _trading_client