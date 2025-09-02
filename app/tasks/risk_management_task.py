import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import TrailingStopOrderRequest, LimitOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from .celery_app import celery_instance
from app.config import settings
from app.redis_client import redis_client
from app.trading.alpaca_client import get_trading_client

@celery_instance.task(name="tasks.monitor_positions_and_risk")
def monitor_positions_and_risk():
    trading_client = get_trading_client()
    
    try:
        positions = trading_client.get_all_positions()
        account = trading_client.get_account()

        initial_equity_key = "daily_initial_equity"
        if not redis_client.exists(initial_equity_key):
            redis_client.set(initial_equity_key, account.equity, ex=86400)

        initial_equity = float(redis_client.get(initial_equity_key) or account.equity)
        current_equity = float(account.equity)
        daily_pnl_pct = (current_equity - initial_equity) / initial_equity

        if daily_pnl_pct <= -0.03:
            if not redis_client.get("circuit_breaker_active"):
                logging.critical(f"Circuit breaker activated: {daily_pnl_pct:.2%} daily loss")
                try:
                    trading_client.close_all_positions(cancel_orders=True)
                except Exception as e:
                    logging.error(f"Failed to close all positions: {e}")
                redis_client.set("circuit_breaker_active", "true", ex=86400)
            return

        for p in positions:
            try:
                pos_key = f"position:{p.symbol}"
                if not redis_client.exists(pos_key):
                    continue

                pos_data = redis_client.hgetall(pos_key)
                entry_price = float(pos_data[b'entry_price'])
                stop_loss = float(pos_data[b'stop_loss'])
                risk_per_share = abs(entry_price - stop_loss)
                current_price = float(p.current_price)
                
                if p.side == 'long':
                    pnl_per_share = current_price - entry_price
                    profit_target_1R = entry_price + risk_per_share
                else:
                    pnl_per_share = entry_price - current_price
                    profit_target_1R = entry_price - risk_per_share
                
                current_r_multiple = pnl_per_share / risk_per_share if risk_per_share > 0 else 0

                if b'be_triggered' not in pos_data and current_r_multiple >= 1.0:
                    logging.info(f"Position {p.symbol} reached 1R profit, implementing break-even strategy")
                    
                    try:
                        trading_client.cancel_orders_for(p.symbol)
                        
                        partial_qty = float(p.qty) / 2
                        side_to_close = OrderSide.SELL if p.side == 'long' else OrderSide.BUY
                        
                        partial_close = trading_client.submit_order(
                            order_data=LimitOrderRequest(
                                symbol=p.symbol, 
                                qty=partial_qty, 
                                side=side_to_close, 
                                limit_price=float(p.current_price),
                                time_in_force=TimeInForce.GTD
                            )
                        )
                        
                        if partial_close:
                            trail_percent = 2.0
                            trailing_stop = trading_client.submit_order(
                                order_data=TrailingStopOrderRequest(
                                    symbol=p.symbol, 
                                    qty=partial_qty, 
                                    side=side_to_close, 
                                    trail_percent=str(trail_percent),
                                    time_in_force=TimeInForce.GTC
                                )
                            )
                            
                            if trailing_stop:
                                redis_client.hset(pos_key, "be_triggered", "true")
                                redis_client.hset(pos_key, "partial_close_order", partial_close.id)
                                redis_client.hset(pos_key, "trailing_stop_order", trailing_stop.id)
                                logging.info(f"Break-even strategy implemented for {p.symbol}")
                    
                    except Exception as e:
                        logging.error(f"Failed to implement break-even for {p.symbol}: {e}")

                elif current_r_multiple <= -2.0:
                    logging.warning(f"Position {p.symbol} hit -2R, emergency close")
                    try:
                        trading_client.close_position(p.symbol)
                        redis_client.delete(pos_key)
                    except Exception as e:
                        logging.error(f"Failed emergency close for {p.symbol}: {e}")

            except Exception as e:
                logging.error(f"Error processing position {p.symbol}: {e}")

    except Exception as e:
        logging.error(f"Error in position monitoring: {e}", exc_info=True)

@celery_instance.task(name="tasks.daily_equity_snapshot")
def daily_equity_snapshot():
    try:
        trading_client = get_trading_client()
        account = trading_client.get_account()
        
        redis_client.set("daily_initial_equity", account.equity, ex=86400)
        redis_client.delete("circuit_breaker_active")
        
        logging.info(f"Daily equity snapshot: ${float(account.equity):,.2f}")
        
    except Exception as e:
        logging.error(f"Error in daily equity snapshot: {e}")