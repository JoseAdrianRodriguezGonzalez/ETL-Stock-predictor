import pytest

from collector.normalization.market_data import MarketDataNormalizer


@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "changePercent",
        "vwap",
    ],
)
def test_normalize_null_values(fmp_record: dict, field: str) -> None:
    fmp_record[field] = None

    with pytest.raises(ValueError, match="Null values"):
        MarketDataNormalizer.normalize(fmp_record)
