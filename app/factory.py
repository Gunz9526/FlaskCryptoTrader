import sys
from flask import Flask
from .config import settings
from .logging_config import setup_logging
from .extensions import db, migrate, metrics

def create_app():
    app = Flask(__name__)
    app.config.from_object(settings)

    is_celery_worker = any('celery' in arg for arg in sys.argv)

    if not is_celery_worker:
        setup_logging()

    db.init_app(app)
    migrate.init_app(app, db)
    metrics.init_app(app)

    register_blueprints(app)
    
    from . import models
    
    return app

def register_blueprints(app):
    from .api import create_api_blueprint
    
    api_bp = create_api_blueprint()
    app.register_blueprint(api_bp, url_prefix='/api')