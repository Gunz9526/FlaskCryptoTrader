from flask import request, jsonify

def register_trading_routes(bp):
    @bp.route('/train', methods=['POST'])
    def force_train():
        from app.tasks.ml_task import trigger_model_retraining

        auth_key = request.headers.get('X-Auth-Key')
        if auth_key != 'my-secret-training-key':
            return jsonify({"error": "Unauthorized"}), 401

        task = trigger_model_retraining.delay()
        
        return jsonify({
            "message": "Model retraining task has been triggered.", 
            "task_id": task.id
        }), 202

    @bp.route('/positions', methods=['GET'])
    def get_positions():
        from app.trading.alpaca_client import get_trading_client
        from app.config import settings
        
        try:
            client = get_trading_client()
            positions = client.get_all_positions()
            
            return jsonify({
                'positions': [
                    {
                        'symbol': p.symbol,
                        'qty': float(p.qty),
                        'side': p.side,
                        'market_value': float(p.market_value),
                        'unrealized_pl': float(p.unrealized_pl)
                    } for p in positions
                ]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @bp.route('/trigger-trade-cycle', methods=['POST'])
    def trigger_trade_cycle():
        from app.tasks.trading_task import execute_trade_cycle
        from app.tasks import SUPPORTED_SYMBOLS

        auth_key = request.headers.get('X-Auth-Key')
        if auth_key != 'my-secret-training-key':
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Missing required field: symbol'}), 400
        
        symbol = data['symbol']
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': f'Symbol {symbol} is not supported'}), 400

        try:
            task = execute_trade_cycle.delay(symbol)
            
            return jsonify({
                "message": f"AI trading cycle for {symbol} has been triggered.",
                "task_id": task.id
            }), 202
        except Exception as e:
            return jsonify({'error': str(e)}), 500
