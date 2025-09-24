from .celery_app import celery_instance
from app.trading.execution_engine import run_trading_cycle_for_symbol


@celery_instance.task(name="tasks.execute_trade_cycle")
def execute_trade_cycle(symbol: str):
    run_trading_cycle_for_symbol(symbol)

