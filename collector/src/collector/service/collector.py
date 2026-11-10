from datetime import timedelta

from collector.normalization.market_data import MarketDataNormalizer
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import (
    MarketDataRepository,
    UpsertResult,
)


class CollectorService:
    def __init__(
        self,
        fmp_client: FMPClient,
        repository: MarketDataRepository,
    ) -> None:
        self.fmp_client = fmp_client
        self.repository = repository

    def collect_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> UpsertResult:
        # 1. Extraer datos del proveedor
        records = self.fmp_client.get_historical_prices(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # 2. Normalizar registros
        normalized_records = MarketDataNormalizer.normalize_many(records)

        # 3. Persistir en MongoDB
        return self.repository.upsert_many(normalized_records)

    def collect_incremental(
        self,
        symbol: str,
        overlap_days: int = 3,
    ) -> UpsertResult:

        latest_date = self.repository.get_latest_date(symbol)

        if latest_date is None:
            raise ValueError(
                f"No historical data found for {symbol}. "
                "Run historical collection first."
            )

        start_date = latest_date - timedelta(days=overlap_days)

        return self.collect_historical(
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=None,
        )
