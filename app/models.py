from app.extensions import db
from sqlalchemy.dialects.postgresql import BIGINT

class MarketData(db.Model):
    __tablename__ = 'market_data'

    symbol = db.Column(db.String, nullable=False, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, primary_key=True, index=True)
    timeframe = db.Column(db.String, nullable=False, primary_key=True)
    
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(BIGINT, nullable=False)
    
    trade_count = db.Column(db.Integer, nullable=True)
    vwap = db.Column(db.Float, nullable=True)


    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}')>"