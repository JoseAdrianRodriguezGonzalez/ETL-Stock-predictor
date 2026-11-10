

import json
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from collector.normalization.market_data import MarketDataNormalizer
from collector.repositories.market_data import MarketDataRepository
from collector.snapshot import SnapshotExporter


@pytest.fixture
def repository(fmp_record: dict) -> MagicMock:
    first = fmp_record.copy()
    second = fmp_record.copy()

    first["date"] = "2026-11-06"
    second["date"] = "2026-11-09"

    records = MarketDataNormalizer.normalize_many([second, first])

    mock = MagicMock(spec=MarketDataRepository)
    mock.get_records.return_value = records

    return mock


def test_export_snapshot(repository: MagicMock, tmp_path) -> None:
    exporter = SnapshotExporter(repository)

    result = exporter.export(
        symbol="^GSPC",
        output_dir=str(tmp_path),
        dataset="sp500_historical",
        source="financial_modeling_prep",
    )

    assert result.records == 2
    assert result.parquet_path.endswith("historical.parquet")
    assert result.manifest_path.endswith("manifest.json")

    parquet_table = pq.read_table(result.parquet_path)
    manifest = json.loads(
        open(result.manifest_path, encoding="utf-8").read()
    )

    assert parquet_table.num_rows == 2
    assert parquet_table.schema == exporter.SCHEMA

    assert manifest["dataset"] == "sp500_historical"
    assert manifest["symbol"] == "^GSPC"
    assert manifest["records"] == parquet_table.num_rows
    assert manifest["period"]["start"] == "2026-11-06"
    assert manifest["period"]["end"] == "2026-11-09"


def test_export_snapshot_empty_dataset(tmp_path) -> None:
    repository = MagicMock(spec=MarketDataRepository)
    repository.get_records.return_value = []

    exporter = SnapshotExporter(repository)

    with pytest.raises(ValueError, match="No market data found"):
        exporter.export(
            symbol="^GSPC",
            output_dir=str(tmp_path),
            dataset="sp500_historical",
            source="financial_modeling_prep",
        )


def test_export_snapshot_unexpected_symbol(
    repository: MagicMock,
    tmp_path,
) -> None:
    exporter = SnapshotExporter(repository)

    with pytest.raises(ValueError, match="Unexpected symbol"):
        exporter.export(
            symbol="AAPL",
            output_dir=str(tmp_path),
            dataset="sp500_historical",
            source="financial_modeling_prep",
        )
