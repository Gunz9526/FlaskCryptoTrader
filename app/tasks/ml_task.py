import datetime
import pandas as pd
import logging
from app.ml.train.lgbm import LGBMTrainer
from app.ml.train.xgb import XGBTrainer
from app.ml.train.catboost import CatBoostTrainer
import numpy as np
from .celery_app import celery_instance
from sqlalchemy import select, text
from app.extensions import db
from app.tasks import SUPPORTED_SYMBOLS



@celery_instance.task(name="tasks.trigger_model_retraining")
def trigger_model_retraining():
    logging.info("Starting comprehensive model retraining cycle")
    
    for symbol in SUPPORTED_SYMBOLS:
        try:
            stmt_15m = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '15m' ORDER BY timestamp ASC")
            stmt_1h = text("SELECT * FROM market_data WHERE symbol = :symbol AND timeframe = '1h' ORDER BY timestamp ASC")
            
            df_15m = pd.read_sql(stmt_15m, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')
            df_1h = pd.read_sql(stmt_1h, db.session.connection(), params={'symbol': symbol}, index_col='timestamp')

            if len(df_15m) < 2000:
                logging.warning(f"[{symbol}] Insufficient data. Skipping.")
                continue

            for model_type in ['ranging', 'trending']:
                trainers = [
                    LGBMTrainer(symbol, model_type),
                    XGBTrainer(symbol, model_type),
                    CatBoostTrainer(symbol, model_type),
                ]
                for trainer in trainers:
                    trainer.train(df_15m.copy(), df_1h.copy())

        except Exception as e:
            logging.error(f"Error during training cycle for {symbol}: {e}", exc_info=True)
            
    logging.info("Model retraining cycle completed successfully")