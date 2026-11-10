import os

from collector.config import PROJECT_ROOT, get_settings
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import MarketDataRepository
from collector.service.collector import CollectorService
from collector.snapshot import SnapshotExporter
from collector.storage.mongo import MongoConnection
from collector.versioning import DVCVersioner


def main() -> None:
    settings = get_settings()

    symbol = "^GSPC"
    output_dir = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
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

        latest_date = repository.get_latest_date(symbol)

        if latest_date is None:
            print("No historical data found. Starting initial collection.")

            result = collector.collect_historical(
                symbol=symbol,
                start_date="2010-01-04",
                end_date=None,
            )
        else:
            print(f"Existing data found. Latest date: {latest_date}")
            print("Starting incremental collection.")

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

        versioner = DVCVersioner(
            repo_root=PROJECT_ROOT,
        )

        versioner.version_dataset(
            dataset_path=os.path.join(
                "data",
                "processed",
                "sp500",
            )
        )

        print("\nDataset registered in DVC")

        versioner.push()

        print("Dataset pushed to DVC remote")
if __name__ == "__main__":
    main()
