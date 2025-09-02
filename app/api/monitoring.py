from flask import jsonify
import os

def register_monitoring_routes(bp):
    @bp.route('/system-status', methods=['GET'])
    def system_status():
        from app.models import MarketData
        from app.redis_client import redis_client
        from app.extensions import db
        
        try:
            data_counts = {
                '15m': MarketData.query.filter_by(timeframe='15m').count(),
                '1h': MarketData.query.filter_by(timeframe='1h').count()
            }
            
            supported_symbols = ['BTC/USD', 'ETH/USD']
            regime = {
                symbol: (redis_client.get(f"market_regime:{symbol}") or b'unknown').decode()
                for symbol in supported_symbols
            }
            sentiment = redis_client.get("news_sentiment_score")
            circuit_breaker = redis_client.get("circuit_breaker_active")
            
            model_types = ['ranging_artifacts', 'trending_artifacts']
            models_available = {
                symbol: {
                    model_type: os.path.exists(f'./models/{symbol.lower().replace("/", "_")}/{model_type}.pkl')
                    for model_type in model_types
                }
                for symbol in supported_symbols
            }
            
            return jsonify({
                'data_counts': data_counts,
                'market_regime': regime,
                'sentiment_score': float(sentiment) if sentiment else 0.0,
                'circuit_breaker_active': bool(circuit_breaker),
                'models_available': models_available,
                'models_ready': all(all(model_types.values()) for model_types in models_available.values())
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500