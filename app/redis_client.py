import redis
from app.config import settings

redis_client = redis.from_url(settings.REDIS_URL.unicode_string())