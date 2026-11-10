

import json
from datetime import UTC, datetime

import pytest

from collector.manifest import ManifestGenerator, ManifestWriter
from collector.normalization.market_data import MarketDataNormalizer


def test_generate_manifest(fmp_record: dict) -> None:
    records = MarketDataNormalizer.normalize_many([fmp_record])

    collected_at = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)

    manifest = ManifestGenerator.generate(
        records,
        dataset="sp500_historical",
        source="financial_modeling_prep",
        collected_at=collected_at,
    )

    assert manifest.dataset == "sp500_historical"
    assert manifest.source == "financial_modeling_prep"
    assert manifest.symbol == "^GSPC"
    assert manifest.records == 1
    assert manifest.period.start.isoformat() == "2026-11-09"
    assert manifest.period.end.isoformat() == "2026-11-09"
    assert manifest.frequency == "1d"
    assert manifest.format == "parquet"
    assert manifest.schema_version == "1.0.0"


def test_generate_manifest_multiple_records(fmp_record: dict) -> None:
    first = fmp_record.copy()
    second = fmp_record.copy()

    first["date"] = "2026-11-06"
    second["date"] = "2026-11-09"

    records = MarketDataNormalizer.normalize_many([first, second])

    manifest = ManifestGenerator.generate(
        records,
        dataset="sp500_historical",
        source="financial_modeling_prep",
    )

    assert manifest.records == 2
    assert manifest.period.start.isoformat() == "2026-11-06"
    assert manifest.period.end.isoformat() == "2026-11-09"


def test_generate_manifest_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        ManifestGenerator.generate(
            [],
            dataset="sp500_historical",
            source="financial_modeling_prep",
        )


def test_generate_manifest_multiple_symbols(fmp_record: dict) -> None:
    first = fmp_record.copy()
    second = fmp_record.copy()
    second["symbol"] = "AAPL"

    records = MarketDataNormalizer.normalize_many([first, second])

    with pytest.raises(ValueError, match="same symbol"):
        ManifestGenerator.generate(
            records,
            dataset="market_data",
            source="financial_modeling_prep",
        )


def test_generate_manifest_naive_datetime(fmp_record: dict) -> None:
    records = MarketDataNormalizer.normalize_many([fmp_record])

    with pytest.raises(ValueError, match="timezone-aware"):
        ManifestGenerator.generate(
            records,
            dataset="sp500_historical",
            source="financial_modeling_prep",
            collected_at=datetime(2026, 11, 10, 12, 0),
        )


def test_write_manifest(tmp_path, fmp_record: dict) -> None:
    records = MarketDataNormalizer.normalize_many([fmp_record])

    manifest = ManifestGenerator.generate(
        records,
        dataset="sp500_historical",
        source="financial_modeling_prep",
        collected_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
    )

    output_path = tmp_path / "dataset" / "manifest.json"

    ManifestWriter.write(manifest, output_path)

    assert output_path.exists()

    content = json.loads(output_path.read_text(encoding="utf-8"))

    assert content["dataset"] == "sp500_historical"
    assert content["symbol"] == "^GSPC"
    assert content["records"] == 1
    assert content["collected_at"] == "2026-11-10T12:00:00Z"
    assert content["period"]["start"] == "2026-11-09"
