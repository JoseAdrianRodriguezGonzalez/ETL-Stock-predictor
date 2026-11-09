from datetime import date
import pytest
from collector.normalization.market_data import (MarketDataNormalizer)
@pytest.fixture
def fmp_record() -> dict:
    return {
        "symbol": "^GSPC",
        "date": "2026-11-09",
        "open": 7422.24,
        "high": 7577.39,
        "low": 7422.24,
        "close": 7556.16,
        "volume": 2598689000,
        "change": 133.92,
        "changePercent": 1.80431,
        "vwap": 7518.6,
    }


def test_normalize_success(fmp_record: dict) -> None:
    result = MarketDataNormalizer.normalize(fmp_record)

    assert result["symbol"] == "^GSPC"
    assert result["date"] == date(2026, 11, 9)
    assert result["open"] == 7422.24
    assert result["high"] == 7577.39
    assert result["low"] == 7422.24
    assert result["close"] == 7556.16
    assert result["volume"] == 2598689000
    assert result["change"] == 133.92
    assert result["change_percent"] == 1.80431
    assert result["vwap"] == 7518.6

    assert "changePercent" not in result


def test_normalize_numeric_types(fmp_record: dict) -> None:
    fmp_record.update(
        {
            "open": "7422.24",
            "high": "7577.39",
            "low": "7422.24",
            "close": "7556.16",
            "volume": "2598689000",
            "change": "133.92",
            "changePercent": "1.80431",
            "vwap": "7518.6",
        }
    )

    result = MarketDataNormalizer.normalize(fmp_record)

    for field in (
        "open",
        "high",
        "low",
        "close",
        "change",
        "change_percent",
        "vwap",
    ):
        assert isinstance(result[field], float)

    assert isinstance(result["volume"], int)


def test_normalize_missing_fields(fmp_record: dict) -> None:
    del fmp_record["close"]

    with pytest.raises(
        ValueError,
        match="Missing required market data fields",
    ):
        MarketDataNormalizer.normalize(fmp_record)


@pytest.mark.parametrize(
    "invalid_date",
    [
        "09-11-2026",
        "2026/11/09",
        "invalid-date",
    ],
)
def test_normalize_invalid_date(
    fmp_record: dict,
    invalid_date: str,
) -> None:
    fmp_record["date"] = invalid_date

    with pytest.raises(ValueError, match="Invalid market data record"):
        MarketDataNormalizer.normalize(fmp_record)


@pytest.mark.parametrize(
    "field",
    [
        "open",
        "high",
        "low",
        "close",
        "change",
        "changePercent",
        "vwap",
        "volume",
    ],
)
def test_normalize_invalid_numeric_value(
    fmp_record: dict,
    field: str,
) -> None:
    fmp_record[field] = "invalid"

    with pytest.raises(ValueError, match="Invalid market data record"):
        MarketDataNormalizer.normalize(fmp_record)


def test_normalize_empty_symbol(fmp_record: dict) -> None:
    fmp_record["symbol"] = " "

    with pytest.raises(ValueError, match="symbol cannot be empty"):
        MarketDataNormalizer.normalize(fmp_record)


def test_normalize_negative_volume(fmp_record: dict) -> None:
    fmp_record["volume"] = -1

    with pytest.raises(ValueError, match="volume cannot be negative"):
        MarketDataNormalizer.normalize(fmp_record)


def test_normalize_invalid_high(fmp_record: dict) -> None:
    fmp_record["high"] = 7400.0

    with pytest.raises(ValueError, match="high must be"):
        MarketDataNormalizer.normalize(fmp_record)


def test_normalize_invalid_low(fmp_record: dict) -> None:
    fmp_record["low"] = 7600.0

    with pytest.raises(ValueError, match="low must be"):
        MarketDataNormalizer.normalize(fmp_record)


def test_normalize_many(fmp_record: dict) -> None:
    second_record = {
        **fmp_record,
        "date": "2026-11-06",
        "close": 7500.0,
        "high": 7577.39,
    }

    results = MarketDataNormalizer.normalize_many(
        [fmp_record, second_record]
    )

    assert len(results) == 2
    assert results[0]["date"] == date(2026, 11, 9)
    assert results[1]["date"] == date(2026, 11, 6)


def test_normalize_many_fails_on_invalid_record(
    fmp_record: dict,
) -> None:
    invalid_record = {**fmp_record}
    del invalid_record["close"]

    with pytest.raises(ValueError):
        MarketDataNormalizer.normalize_many(
            [fmp_record, invalid_record]
        )
