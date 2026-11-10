import os
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from collector.manifest import (
    DatasetManifest,
    ManifestGenerator,
    ManifestWriter,
)
from collector.preprocessing.market_data import MarketDataPreprocessor
from collector.repositories.market_data import MarketDataRepository
from collector.validation.market_data import MarketDataValidator


@dataclass(frozen=True)
class SnapshotResult:
    parquet_path: str
    manifest_path: str
    records: int
    manifest: DatasetManifest


class SnapshotExporter:
    SCHEMA = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("date", pa.date32(), nullable=False),
            pa.field("open", pa.float64(), nullable=False),
            pa.field("high", pa.float64(), nullable=False),
            pa.field("low", pa.float64(), nullable=False),
            pa.field("close", pa.float64(), nullable=False),
            pa.field("volume", pa.int64(), nullable=False),
            pa.field("change", pa.float64(), nullable=False),
            pa.field("change_percent", pa.float64(), nullable=False),
            pa.field("vwap", pa.float64(), nullable=False),
        ]
    )

    def __init__(self, repository: MarketDataRepository) -> None:
        self.repository = repository

    def export(
        self,
        symbol: str,
        output_dir: str,
        *,
        dataset: str,
        source: str,
        frequency: str = "1d",
        schema_version: str = "1.0.0",
    ) -> SnapshotResult:
        records = self.repository.get_records(symbol)

        if not records:
            raise ValueError(f"No market data found for symbol {symbol}")

        processed_records = MarketDataPreprocessor.sort_by_date(records)

        MarketDataValidator.validate(
            processed_records,
            expected_symbol=symbol,
        )

        table = pa.Table.from_pylist(
            [dict(record) for record in processed_records],
            schema=self.SCHEMA,
        )

        os.makedirs(output_dir, exist_ok=True)

        parquet_path = os.path.join(output_dir, "historical.parquet")
        manifest_path = os.path.join(output_dir, "manifest.json")

        pq.write_table(
            table,
            parquet_path,
            compression="zstd",
        )

        manifest = ManifestGenerator.generate(
            processed_records,
            dataset=dataset,
            source=source,
            frequency=frequency,
            file_format="parquet",
            schema_version=schema_version,
        )

        ManifestWriter.write(manifest, manifest_path)

        return SnapshotResult(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            records=len(processed_records),
            manifest=manifest,
        )
