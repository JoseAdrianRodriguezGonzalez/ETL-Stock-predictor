

from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SP500 Forecasting API"
    app_version: str = "0.1.0"

    # Triton
    triton_url: str = "triton:8000"
    triton_model_name: str = "sp500_forecasting"
    triton_model_version: str = "1"
    request_timeout: float = 10.0

    # MongoDB
    mongo_uri: str | None = None
    mongo_user: str = "sp500"
    mongo_password: str | None = None
    mongo_database: str = "market_data"
    mongo_host: str = "mongodb"
    mongo_port: int = 27017

    # Inference
    scaler_path: str = "../deployment/assets/scaler.joblib"
    sequence_length: int = 30

    @property
    def mongodb_uri(self) -> str:
        if self.mongo_uri:
            return self.mongo_uri

        if not self.mongo_password:
            raise ValueError("MONGO_PASSWORD is required")

        username = quote_plus(self.mongo_user)
        password = quote_plus(self.mongo_password)

        return (
            f"mongodb://{username}:{password}"
            f"@{self.mongo_host}:{self.mongo_port}"
            f"/{self.mongo_database}?authSource=admin"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

