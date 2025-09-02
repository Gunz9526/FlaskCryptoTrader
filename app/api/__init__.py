from flask import Blueprint

def create_api_blueprint():
    api_bp = Blueprint('api', __name__)
    
    from .data import register_data_routes
    from .trading import register_trading_routes
    from .monitoring import register_monitoring_routes
    from .health import register_health_routes
    
    register_data_routes(api_bp)
    register_trading_routes(api_bp)
    register_monitoring_routes(api_bp)
    register_health_routes(api_bp)
    
    return api_bp