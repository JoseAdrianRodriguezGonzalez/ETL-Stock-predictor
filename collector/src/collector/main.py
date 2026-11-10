from collector.config import get_settings
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import MarketDataRepository
from collector.services.collector import CollectorService
from collector.storage.mongo import MongoConnection


def main() -> None:
    settings = get_settings()

    with (FMPClient(settings) as fmp,MongoConnection(settings) as mongo):
        repository = MarketDataRepository(mongo.database)
        repository.ensure_indexes()

        collector = CollectorService(
            fmp_client=fmp,
            repository=repository,
        )

        result = collector.collect_historical(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2026-11-09",
        )

        print(f"Upserted: {result.upserted}")
        print(f"Modified: {result.modified}")
        print(f"Latest date: {repository.get_latest_date('^GSPC')}")


if __name__ == "__main__":
    main()

