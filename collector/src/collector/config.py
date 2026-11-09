import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote_plus
@dataclass(frozen=True)
class Settings:
    app_env:str 
    log_level:str
    #fmp 
    fmp_api_key:str 
    fmp_base_url:str
    fmp_timeout:float 
    fmp_max_retries:int 

    #Mongo 
    mongo_uri:str | None 
    mongo_user: str
    mongo_password:str | None 
    mongo_database: str
    mongo_host: str
    mongo_port: int 
    
    @property
    def mongodb_uri(self)->str:
        if self.mongo_uri:
            return self.mongo_uri
        if not self.mongo_password:
            raise ValueError("Debe de tener una contraseña")
        username=quote_plus(self.mongo_user)
        password=quote_plus(self.mongo_password)
        return (
            f"mongodb://{username}:{password}"
                f"@{self.mongo_host}:{self.mongo_port}"
                f"/{self.mongo_database}?authSource=admin"
        )
@lru_cache
def get_settings()->Settings:
    api_key=os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY is required")
    timeout = float(os.getenv("FMP_TIMEOUT", "30"))
    retries = int(os.getenv("FMP_MAX_RETRIES", "3"))
    port = int(os.getenv("MONGO_PORT", "27017"))

    if timeout <= 0:
        raise ValueError("FMP_TIMEOUT debe ser mayor que cero")

    if retries < 0:
        raise ValueError("FMP_MAX_RETRIES no puede ser negativo")

    if not 1 <= port <= 65535:
        raise ValueError("MONGO_PORT debe estar entre 1 y 65535")
    return Settings(
        app_env=os.getenv("APP_ENV","development"),
        log_level=os.getenv("LOG_LEVEL","INFO"),
        fmp_api_key=api_key,
        fmp_base_url=os.getenv(
            "FMP_BASE_URL",
            "https://financialmodelingprep.com/stable",
        ),
        fmp_timeout=timeout,
        fmp_max_retries=retries,

        mongo_uri=os.getenv("MONGO_URI"),
        mongo_user=os.getenv("MONGO_USER", "sp500"),
        mongo_password=os.getenv("MONGO_PASSWORD"),
        mongo_database=os.getenv("MONGO_DATABASE", "market_data"),
        mongo_host=os.getenv("MONGO_HOST", "localhost"),
        mongo_port=port,
    )
