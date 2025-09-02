import time
import pandas as pd
import logging
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pytz
from .celery_app import celery_instance
from app.redis_client  import redis_client
from app.config import settings
from app.models import MarketData
from app.extensions import db
from app.metrics import DATA_PIPELINE_ROWS_COUNTER
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func


UTC = pytz.UTC

def _bulk_insert_data(df: pd.DataFrame, symbol: str, timeframe: str):
    if df.empty:
        logging.info(f"No data to process for {symbol} {timeframe}.")
        return 0

    df = df.reset_index()

    records_to_add = []
    for _, row in df.iterrows():
        ts = row['timestamp']
        if isinstance(ts, pd.Timestamp):
            ts_py = ts.to_pydatetime()
        else:
            ts_py = ts
        ts_py = _ensure_utc(ts_py).replace(second=0, microsecond=0)

        records_to_add.append(dict(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts_py,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
            trade_count=int(row.get('trade_count', 0)),
            vwap=float(row.get('vwap', row['close']))
        ))

    if not records_to_add:
        return 0

    stmt = insert(MarketData).values(records_to_add)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['symbol', 'timestamp', 'timeframe']
    ).returning(MarketData.timestamp)

    try:
        result = db.session.execute(stmt)
        inserted_rows = result.fetchall()
        inserted_count = len(inserted_rows)
        db.session.commit()

        if inserted_count > 0:
            DATA_PIPELINE_ROWS_COUNTER.labels(symbol=symbol, timeframe=timeframe).inc(inserted_count)
            logging.info(f"Saved {inserted_count} new {timeframe} bars for {symbol}")
        else:
            logging.info(f"No new {timeframe} data to save for {symbol} (all duplicates).")

        return inserted_count

    except Exception as e:
        logging.error(f"Error during bulk insert for {symbol} {timeframe}: {e}", exc_info=True)
        db.session.rollback()
        return 0

@celery_instance.task(name="tasks.collect_15m_data")
def collect_15m_data(symbol: str):
    logging.info(f"Starting 15m data collection for {symbol}")
    
    try:
        client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(hours=48)
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time + timedelta(minutes=15)
        )
        
        bars = client.get_crypto_bars(request_params)
        
        if bars.df.empty:
            logging.warning(f"No 15m data returned for {symbol}")
            return
            
        df = bars.df.loc[symbol]
        if df.empty:
            logging.warning(f"No 15m data returned for {symbol} after symbol extraction")
            return
        logging.info(f"Retrieved {len(df)} 15m bars for {symbol}")
        
        _bulk_insert_data(df, symbol, '15m')
        
    except Exception as e:
        logging.error(f"Error collecting 15m data for {symbol}: {e}", exc_info=True)
        db.session.rollback()

@celery_instance.task(name="tasks.collect_1h_data")
def collect_1h_data(symbol: str):
    logging.info(f"Starting 1h data collection for {symbol}")
    
    try:
        client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=30)
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time + timedelta(hours=1)
        )
        
        bars = client.get_crypto_bars(request_params)
        
        if bars.df.empty:
            logging.warning(f"No 1h data returned for {symbol}")
            return
            
        df = bars.df.loc[symbol]
        if df.empty:
            logging.warning(f"No 1h data returned for {symbol} after symbol extraction")
            return
        logging.info(f"Retrieved {len(df)} 1h bars for {symbol}")
        
        _bulk_insert_data(df, symbol, '1h')
        
    except Exception as e:
        logging.error(f"Error collecting 1h data for {symbol}: {e}", exc_info=True)
        db.session.rollback()

@celery_instance.task(name="tasks.initialize_historical_data")
def initialize_historical_data(symbol: str):
    logging.info(f"Initializing historical data for {symbol}")
    
    try:
        client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        
        end_time = datetime.now(pytz.UTC)
        start_time_15m = end_time - timedelta(days=30)
        start_time_1h = end_time - timedelta(days=365)

        for timeframe, start_time, tf_obj in [
            ('15m', start_time_15m, TimeFrame(15, TimeFrameUnit.Minute)),
            ('1h', start_time_1h, TimeFrame.Hour)
        ]:
            logging.info(f"Collecting {timeframe} data for {symbol} from {start_time} to {end_time}")
            
            if timeframe == '15m':
                exclusive_end = end_time + timedelta(minutes=15)
            else:
                exclusive_end = end_time + timedelta(hours=1)
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf_obj,
                start=start_time,
                end=exclusive_end
            )
            
            bars = client.get_crypto_bars(request_params)
            
            if not bars.df.empty:
                df = bars.df.loc[symbol]
                if not df.empty:
                    count = _bulk_insert_data(df, symbol, timeframe)
                    logging.info(f"Saved {count} new {timeframe} bars for {symbol}")
                else:
                    logging.warning(f"No {timeframe} data available for {symbol} after symbol extraction")
            else:
                logging.warning(f"No {timeframe} data available for {symbol}")
                
    except Exception as e:
        logging.error(f"Error initializing historical data for {symbol}: {e}", exc_info=True)
        db.session.rollback()

@celery_instance.task(name="tasks.collect_bulk_historical_data", bind=True)
def collect_bulk_historical_data(self, symbol: str, years: int = 2):
    logging.info(f"Starting bulk historical data collection for {symbol} ({years} years)")
    
    try:
        client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=years * 365)
        
        total_saved = {'15m': 0, '1h': 0}
        
        timeframe_configs = [
            {
                'name': '15m',
                'tf_obj': TimeFrame(15, TimeFrameUnit.Minute),
                'chunk_days': 7,
                'description': '15분봉'
            },
            {
                'name': '1h', 
                'tf_obj': TimeFrame.Hour,
                'chunk_days': 30,
                'description': '1시간봉'
            }
        ]
        
        for config in timeframe_configs:
            timeframe = config['name']
            tf_obj = config['tf_obj']
            chunk_days = config['chunk_days']
            
            logging.info(f"Collecting {config['description']} data for {symbol}")
            
            current_start = start_time
            chunk_count = 0
            
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=chunk_days), end_time)
                chunk_count += 1
                
                try:
                    logging.info(f"Chunk {chunk_count}: {current_start.date()} to {current_end.date()}")
                    
                    if timeframe == '15m':
                        exclusive_end = current_end + timedelta(minutes=15)
                    else:
                        exclusive_end = current_end + timedelta(hours=1)
                    
                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=tf_obj,
                        start=current_start,
                        end=exclusive_end
                    )
                    
                    bars = client.get_crypto_bars(request_params)
                    
                    if not bars.df.empty:
                        df = bars.df.loc[symbol]
                        if not df.empty:
                            saved_count = _bulk_insert_data(df, symbol, timeframe)
                            
                            if saved_count > 0:
                                total_saved[timeframe] += saved_count
                            
                            progress = min(100, int(((current_end - start_time).total_seconds() / (end_time - start_time).total_seconds()) * 100))
                            self.update_state(
                                state='PROGRESS',
                                meta={
                                    'current_timeframe': timeframe,
                                    'progress': progress,
                                    'saved_15m': total_saved['15m'],
                                    'saved_1h': total_saved['1h'],
                                    'current_chunk': chunk_count
                                }
                            )
                        else:
                            logging.warning(f"No {timeframe} data for chunk {chunk_count} after symbol extraction")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logging.error(f"Error in chunk {chunk_count}: {e}")
                    time.sleep(2)
                
                current_start = current_end
            
            logging.info(f"Completed {config['description']} collection: {total_saved[timeframe]} bars")
        
        final_result = {
            'status': 'completed',
            'symbol': symbol,
            'years': years,
            'total_saved_15m': total_saved['15m'],
            'total_saved_1h': total_saved['1h'],
            'total_bars': sum(total_saved.values())
        }
        
        logging.info(f"Bulk collection completed: {final_result}")
        return final_result
        
    except Exception as e:
        logging.error(f"Error in bulk historical data collection: {e}", exc_info=True)
        db.session.rollback()
        raise

def _ensure_utc(dt: datetime) -> datetime:
    if dt is None:
        return dt
    if dt.tzinfo is None:
        return UTC.localize(dt)
    return dt.astimezone(UTC)

def _align_boundary(dt: datetime, interval_minutes: int, mode: str) -> datetime:
    dt = _ensure_utc(dt).replace(second=0, microsecond=0)
    m = (dt.minute // interval_minutes) * interval_minutes
    floor_dt = dt.replace(minute=m)
    if mode == 'floor' or dt.minute % interval_minutes == 0:
        return floor_dt
    return floor_dt + timedelta(minutes=interval_minutes)

def _add_empty_gap_to_cache(symbol: str, timeframe: str, gap_start: datetime, gap_end: datetime):
    key = f"gaps:empty:{symbol.replace('/', '_')}:{timeframe}"
    value = f"{gap_start.isoformat()}|{gap_end.isoformat()}"
    redis_client.sadd(key, value)
    redis_client.expire(key, timedelta(days=7))
    logging.info(f"Cached empty gap: [{gap_start}, {gap_end}) for {symbol} {timeframe}")

def _get_empty_gaps_from_cache(symbol: str, timeframe: str) -> set:
    key = f"gaps:empty:{symbol.replace('/', '_')}:{timeframe}"
    empty_gaps_raw = redis_client.smembers(key)
    empty_gaps = set()
    for raw_gap in empty_gaps_raw:
        try:
            start_str, end_str = raw_gap.decode('utf-8').split('|')
            empty_gaps.add((datetime.fromisoformat(start_str), datetime.fromisoformat(end_str)))
        except (ValueError, TypeError):
            continue
    return empty_gaps

@celery_instance.task(name="tasks.collect_missing_data_ranges")
def collect_missing_data_ranges(symbol: str):
    logging.info(f"Checking for missing data ranges for {symbol}")
    try:
        client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        end_time = _ensure_utc(datetime.now(pytz.UTC))
        start_time = end_time - timedelta(days=730)

        for timeframe, tf_obj, interval_minutes, chunk_days in [
            ('15m', TimeFrame(15, TimeFrameUnit.Minute), 15, 7),
            ('1h', TimeFrame.Hour, 60, 30)
        ]:
            expected_interval = timedelta(minutes=interval_minutes)

            aligned_start = _align_boundary(start_time, interval_minutes, 'ceil')
            aligned_end = _align_boundary(end_time, interval_minutes, 'floor')
            logging.info(f"Processing gaps for {symbol} {timeframe} from {aligned_start} to {aligned_end}")

            gaps = find_data_gaps(symbol, timeframe, aligned_start, aligned_end)
            if not gaps:
                logging.info(f"No data gaps found for {symbol} {timeframe}.")
                continue

            logging.info(f"Found {len(gaps)} gaps to fill for {symbol} {timeframe}.")

            total_inserted = 0
            for gap_start, gap_end in gaps:
                current_start = gap_start
                final_end_inclusive = gap_end

                while current_start <= final_end_inclusive:
                    chunk_last_bar = min(current_start + timedelta(days=chunk_days) - expected_interval,
                                         final_end_inclusive)

                    request_start = current_start
                    request_end_inclusive = chunk_last_bar

                    logging.info(f"Filling gap chunk: [{request_start} ~ {request_end_inclusive}] ({timeframe})") 

                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=tf_obj,
                        start=request_start,
                        end=request_end_inclusive
                    )
                    bars = client.get_crypto_bars(request_params)
                    
                    logging.debug(f"API response for chunk [{request_start}, {request_end_inclusive}]: bars.df.shape={bars.df.shape}, empty={bars.df.empty}")
                    
                    if bars.df.empty:
                        logging.info(f"No bars returned for this chunk. Caching as an empty gap.")
                        _add_empty_gap_to_cache(symbol, timeframe, request_start, request_end_inclusive)
                    else:
                        df = bars.df.loc[symbol]
                        logging.debug(f"Extracted df for {symbol}: shape={df.shape}, empty={df.empty}")
                        if df.empty:
                            logging.info(f"No bars returned for this chunk. Caching as an empty gap.")
                            _add_empty_gap_to_cache(symbol, timeframe, request_start, request_end_inclusive)
                        else:
                            existing_timestamps = db.session.query(MarketData.timestamp).filter(
                                MarketData.symbol == symbol,
                                MarketData.timeframe == timeframe,
                                MarketData.timestamp.in_(df.index.tolist())
                            ).all()
                            if existing_timestamps:
                                logging.warning(f"Potential duplicates found for timestamps: {[ts[0] for ts in existing_timestamps]}")
                            
                            inserted = _bulk_insert_data(df, symbol, timeframe)
                            total_inserted += int(inserted or 0)
                            logging.info(f"Inserted {inserted} bars for chunk [{request_start}, {request_end_inclusive}]")

                    current_start = chunk_last_bar + expected_interval
                    time.sleep(0.5)

            logging.info(f"Completed filling gaps for {symbol} {timeframe}. Inserted total: {total_inserted} bars")

        logging.info(f"Missing data collection completed for {symbol}")
    except Exception as e:
        logging.error(f"Error collecting missing data: {e}", exc_info=True)
        db.session.rollback()


def find_data_gaps(symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> list:
    """
    데이터베이스에서 누락된 시간 구간을 찾습니다.
    """
    interval_minutes = 15 if timeframe == '15m' else 60
    expected_interval = timedelta(minutes=interval_minutes)

    start_time = _align_boundary(start_time, interval_minutes, 'ceil')
    end_time = _align_boundary(end_time, interval_minutes, 'floor')

    data = db.session.query(MarketData.timestamp).filter(
        MarketData.symbol == symbol,
        MarketData.timeframe == timeframe,
        MarketData.timestamp.between(start_time, end_time)
    ).order_by(MarketData.timestamp.asc()).all()

    timestamps = [_ensure_utc(row[0]) for row in data]

    db_gaps = []

    if not timestamps:
        if start_time < end_time:
            db_gaps.append((start_time, end_time))
    else:
        if timestamps[0] > start_time:
            db_gaps.append((start_time, timestamps[0] - expected_interval))

        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]
            expected_next = current + expected_interval
            if next_ts > expected_next:
                db_gaps.append((expected_next, next_ts - expected_interval))

        if timestamps[-1] < end_time:
            db_gaps.append((timestamps[-1] + expected_interval, end_time))

    if not db_gaps:
        return []

    empty_gaps_cache = _get_empty_gaps_from_cache(symbol, timeframe)
    if not empty_gaps_cache:
        return db_gaps

    final_gaps = []
    for gap_start, gap_end in db_gaps:
        is_known_empty = False
        for empty_start, empty_end in empty_gaps_cache:
            if empty_start <= gap_start and gap_end < empty_end:
                is_known_empty = True
                break
        
        if not is_known_empty:
            final_gaps.append((gap_start, gap_end))
        else:
            logging.debug(f"Excluding known empty gap: [{gap_start}, {gap_end}] for {symbol} {timeframe}")

    return final_gaps


@celery_instance.task(name="tasks.check_data_gaps")
def check_data_gaps(symbol: str):
    logging.info(f"Checking for data gaps for {symbol}")
    results = {}

    try:
        for timeframe in ['15m', '1h']:
            date_range = db.session.query(
                func.min(MarketData.timestamp).label('min_date'),
                func.max(MarketData.timestamp).label('max_date')
            ).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            ).first()

            if not date_range or not date_range.min_date or not date_range.max_date:
                logging.warning(f"No data found for {symbol} {timeframe} to check for gaps.")
                results[timeframe] = []
                continue

            start_time = _ensure_utc(date_range.min_date)
            end_time = _ensure_utc(date_range.max_date)

            logging.info(f"Processing gaps for {symbol} {timeframe} from {start_time} to {end_time}")
            gaps = find_data_gaps(symbol, timeframe, start_time, end_time)

            results[timeframe] = [
                {'start': gap_start.isoformat(), 'end': gap_end.isoformat()}
                for gap_start, gap_end in gaps
            ]
            logging.info(f"Found {len(gaps)} gaps for {symbol} {timeframe}.")

        return {
            'status': 'completed',
            'symbol': symbol,
            'gaps': results
        }

    except Exception as e:
        logging.error(f"Error checking data gaps for {symbol}: {e}", exc_info=True)
        raise