import sentry_sdk
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.common.exceptions import APIError

from app.trading.alpaca_client import get_trading_client
from app.config import settings

def execute_bracket_order(symbol: str, qty: float, side: str, stop_loss_price: float, limit_price: float):
    trading_client = get_trading_client()
    
    order_side = OrderSide.BUY if side == 'long' else OrderSide.SELL
    
    order_data = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        limit_price=round(limit_price, 2),
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        stop_loss={'stop_price': round(stop_loss_price, 2)}
    )
    
    try:
        submitted_order = trading_client.submit_order(order_data=order_data)
        print("--- SUBMITTING ORDER ---")
        print(f"Order Data: {order_data.dict()}")
        print(f"Order ID: {submitted_order.id}, Status: {submitted_order.status}")
        print("--- ORDER SUBMITTED ---")
        return submitted_order
    except APIError as e:
        print(f"Error submitting bracket order for {symbol}: {e}. Falling back to separate orders.")
        sentry_sdk.capture_exception(e)
        try:
            parent = trading_client.submit_order(order_data=LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=round(limit_price, 2),
                time_in_force=TimeInForce.GTC
            ))
            opp_side = OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY
            trading_client.submit_order(order_data=LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=opp_side,
                limit_price=round(stop_loss_price, 2),
                time_in_force=TimeInForce.GTC
            ))
            return parent
        except Exception as e2:
            print(f"Fallback order failed for {symbol}: {e2}")
            return None