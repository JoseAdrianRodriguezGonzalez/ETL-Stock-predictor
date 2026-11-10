from collector.config import get_settings
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import MarketDataRepository
from collector.service.collector import CollectorService
from collector.storage.mongo import MongoConnection


def main() -> None:
    settings = get_settings()

    with (
        FMPClient(settings) as fmp,
        MongoConnection(settings) as mongo,
    ):
        repository = MarketDataRepository(mongo.database)
        repository.ensure_indexes()

        collector = CollectorService(
            fmp_client=fmp,
            repository=repository,
        )

        result = collector.collect_incremental(
            symbol="^GSPC",
            overlap_days=3,
        )

        print("Incremental collection completed")
        print(f"Matched: {result.matched}")
        print(f"Modified: {result.modified}")
        print(f"Upserted: {result.upserted}")
        print(f"Latest date: {repository.get_latest_date('^GSPC')}")


if __name__ == "__main__":
    main()
