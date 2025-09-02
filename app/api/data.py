from flask import request, jsonify
import logging

def register_data_routes(bp):
    @bp.route('/collect-historical-data', methods=['POST'])
    def collect_historical_data():
        from app.tasks.data_task import collect_bulk_historical_data
        
        try:
            data = request.get_json() or {}
            symbol = data.get('symbol', 'BTC/USD')
            years = min(data.get('years', 2), 2)
            
            task = collect_bulk_historical_data.delay(symbol, years)
            
            return jsonify({
                'status': 'started',
                'task_id': task.id,
                'symbol': symbol,
                'years': years,
                'message': f'Started collecting {years} years of data for {symbol}'
            }), 202
            
        except Exception as e:
            logging.error(f"Error starting historical data collection: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/collection-status/<task_id>', methods=['GET'])
    def get_collection_status(task_id):
        try:
            from app.tasks.celery_app import celery_instance
            task = celery_instance.AsyncResult(task_id)
            
            return jsonify({
                'task_id': task_id,
                'status': task.status,
                'result': task.result,
                'info': task.info
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @bp.route('/update-sentiment', methods=['GET'])
    def update_sentiment():        
        from app.config import settings
        from app.tasks.sentiment_analysis_task import update_news_sentiment
        try:
            update_news_sentiment.delay()
            return jsonify({
                'status': 'started',
                'message': 'Sentiment data update task started'
            }), 202

        except Exception as e:
            logging.error(f"Error updating sentiment data: {e}")
            return jsonify({'error': str(e)}), 500
        
    @bp.route('/fill-gaps', methods=['POST'])
    def fill_data_gaps():
        from app.tasks.data_task import collect_missing_data_ranges
        
        try:
            data = request.get_json() or {}
            symbol = data.get('symbol', 'BTC/USD')
            
            task = collect_missing_data_ranges.delay(symbol)
            
            return jsonify({
                'status': 'started',
                'task_id': task.id,
                'message': f'Started filling data gaps for {symbol}'
            }), 202
            
        except Exception as e:
            logging.error(f"Error starting data gap filling task: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/check-gaps', methods=['POST'])
    def check_gaps():
        from app.tasks.data_task import check_data_gaps
        
        try:
            data = request.get_json() or {}
            symbol = data.get('symbol', 'BTC/USD')
            
            task = check_data_gaps.delay(symbol)
            
            return jsonify({
                'status': 'started',
                'task_id': task.id,
                'message': f'Started checking data gaps for {symbol}'
            }), 202
            
        except Exception as e:
            logging.error(f"Error starting data gap check task: {e}")
            return jsonify({'error': str(e)}), 500
    