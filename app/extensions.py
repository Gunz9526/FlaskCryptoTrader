from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from prometheus_flask_exporter import PrometheusMetrics

db = SQLAlchemy()
migrate = Migrate()
metrics = PrometheusMetrics.for_app_factory()