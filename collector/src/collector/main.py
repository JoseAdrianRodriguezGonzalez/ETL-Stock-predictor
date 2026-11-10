import os

from collector.config import get_settings
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import MarketDataRepository
from collector.service.collector import CollectorService
from collector.snapshot import SnapshotExporter
from collector.storage.mongo import MongoConnection


def main() -> None:
    settings = get_settings()

    symbol = "^GSPC"
    output_dir = os.path.abspath(
        os.path.join(
            os.getcwd(),
            "data",
            "processed",
            "sp500",
        )
    )

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
            symbol=symbol,
            overlap_days=3,
        )

        print("Incremental collection completed")
        print(f"Matched: {result.matched}")
        print(f"Modified: {result.modified}")
        print(f"Upserted: {result.upserted}")
        print(f"Latest date: {repository.get_latest_date(symbol)}")

        exporter = SnapshotExporter(repository)

        snapshot = exporter.export(
            symbol=symbol,
            output_dir=output_dir,
            dataset="sp500_historical",
            source="financial_modeling_prep",
            frequency="1d",
            schema_version="1.0.0",
        )

        print("\nSnapshot generated successfully")
        print(f"Records: {snapshot.records}")
        print(f"Parquet: {snapshot.parquet_path}")
        print(f"Manifest: {snapshot.manifest_path}")


if __name__ == "__main__":
    main()
