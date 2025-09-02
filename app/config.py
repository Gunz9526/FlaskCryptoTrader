from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    REDIS_URL: RedisDsn

    broker_url: str = Field(alias='REDIS_URL')
    result_backend: str = Field(alias='REDIS_URL')
    task_serializer: str = 'json'
    accept_content: list[str] = ['json']
    result_serializer: str = 'json'
    task_track_started: bool = True
    broker_connection_retry_on_startup: bool = True

    SECRET_KEY: str = 'a_default_secret_key'
    
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_PAPER: bool = True
    GEMINI_API_KEY: str | None = None
    SENTRY_DSN: str | None = None

    RISK_PER_TRADE: float = 0.01

settings = Settings()