from .celery_app import celery_instance
from celery.schedules import crontab

SUPPORTED_SYMBOLS = ['BTC/USD', 'ETH/USD']

beat_schedule = {
    'retrain-models-weekly': {
        'task': 'tasks.trigger_model_retraining', 
        'schedule': crontab(hour=2, minute=30, day_of_week='sunday'),
    },
    'daily-equity-snapshot': {
        'task': 'tasks.daily_equity_snapshot',
        'schedule': crontab(hour=0, minute=5),
    },
    'update-news-sentiment': {
        'task': 'tasks.update_news_sentiment', 
        'schedule': 60.0 * 20,
    },
    'monitor-open-positions': {
        'task': 'tasks.monitor_positions_and_risk', 
        'schedule': 45.0,
    },
}

for symbol in SUPPORTED_SYMBOLS:
    symbol_slug = symbol.replace('/', '_').lower()
    
    beat_schedule[f'collect-15m-data-{symbol_slug}'] = {
        'task': 'tasks.collect_15m_data', 
        'schedule': 60.0 * 15, 
        'args': (symbol,),
        'options': {'countdown': 45}
    }
    beat_schedule[f'collect-1h-data-{symbol_slug}'] = {
        'task': 'tasks.collect_1h_data', 
        'schedule': 60.0 * 60, 
        'args': (symbol,),
        'options': {'countdown': 90}
    }
    beat_schedule[f'update-market-regime-{symbol_slug}'] = {
        'task': 'tasks.update_market_regime', 
        'schedule': 60.0 * 60, 
        'args': (symbol,),
        'options': {'countdown': 150}
    }
    beat_schedule[f'run-trading-cycle-{symbol_slug}'] = {
        'task': 'tasks.execute_trade_cycle', 
        'schedule': 60.0 * 15, 
        'args': (symbol,),
        'options': {'countdown': 240}
    }

celery_instance.conf.beat_schedule = beat_schedule


from .data_task import *
from .trading_task import *
from .market_analysis_task import *
from .ml_task import *
from .sentiment_analysis_task import *
from .risk_management_task import *