import pytest

from collector.normalization.market_data import MarketDataNormalizer
from collector.validation.market_data import (
    MarketDataValidationError,
    MarketDataValidator,
)


def normalize_records(
    fmp_record: dict,
    dates: list[str],
) -> list[dict]:
    records = []

    for record_date in dates:
        record = fmp_record.copy()
        record["date"] = record_date
        records.append(MarketDataNormalizer.normalize(record))

    return records


def test_validate_chronological_records(fmp_record: dict) -> None:
    records = normalize_records(
        fmp_record,
        ["2026-11-05", "2026-11-06", "2026-11-09"],
    )

    MarketDataValidator.validate(records, expected_symbol="^GSPC")


def test_validate_unordered_records(fmp_record: dict) -> None:
    records = normalize_records(
        fmp_record,
        ["2026-11-09", "2026-11-06", "2026-11-05"],
    )

    with pytest.raises(
        MarketDataValidationError,
        match="chronological order",
    ):
        MarketDataValidator.validate(records, expected_symbol="^GSPC")


def test_validate_duplicate_dates(fmp_record: dict) -> None:
    records = normalize_records(
        fmp_record,
        ["2026-11-05", "2026-11-05"],
    )

    with pytest.raises(
        MarketDataValidationError,
        match="Duplicate market data record",
    ):
        MarketDataValidator.validate(records, expected_symbol="^GSPC")


def test_validate_empty_dataset() -> None:
    MarketDataValidator.validate([])
