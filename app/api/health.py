from flask import jsonify

def register_health_routes(bp):
    @bp.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok", "service": "FlaskCryptoTrader"}), 200

    @bp.route('/ready', methods=['GET'])
    def readiness_check():
        from app.extensions import db
        from app.redis_client import redis_client
        from sqlalchemy import text
        try:
            db.session.execute(text('SELECT 1'))
            redis_client.ping()
            return jsonify({"status": "ready"}), 200
        except Exception as e:
            return jsonify({"status": "not ready", "error": str(e)}), 503