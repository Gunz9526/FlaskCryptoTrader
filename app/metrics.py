from prometheus_flask_exporter import Counter, Gauge

AI_PREDICTION_COUNTER = Counter(
    'ai_model_prediction_total', 
    'Total number of predictions from AI model',
    labelnames=['symbol', 'regime', 'signal']
)

MARKET_REGIME_GAUGE = Gauge(
    'market_regime_current', 
    'Current market regime identified by the system',
    labelnames=['symbol']
)

DATA_PIPELINE_ROWS_COUNTER = Counter(
    'data_pipeline_rows_saved_total', 
    'Total number of rows saved by the data pipeline',
    labelnames=['symbol', 'timeframe']
)