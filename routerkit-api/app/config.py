from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    bigcodebench_api_url: str = "http://bigcodebench-api:8000"
    artifacts_dir: str = "/data/artifacts"


settings = Settings()
