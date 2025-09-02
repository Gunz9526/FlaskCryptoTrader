from celery import Celery
from app.config import settings


def make_celery():
    celery = Celery('flask_crypto_trader')

    celery_config = settings.model_dump()
    celery_config['broker_url'] = str(settings.REDIS_URL)
    celery_config['result_backend'] = str(settings.REDIS_URL)
    celery.conf.update(celery_config)

    celery.autodiscover_tasks(['app.tasks'])

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            from app.factory import create_app
            flask_app = create_app()
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery_instance = make_celery()
