from datetime import date
from typing import Any, ClassVar, TypedDict


class MarketData(TypedDict):
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float
    change_percent: float
    vwap: float


class MarketDataNormalizer:
    FIELD_MAPPING: ClassVar[dict[str, str]] = {"changePercent": "change_percent"}
    REQUIRED_FIELDS: ClassVar[set[str]] = {
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
    }

    @classmethod
    def normalize(cls, record: dict[str, Any]) -> MarketData:
        missing = cls.REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Missing required market data fields: {sorted(missing)}")
        null_fields = [
            field for field in cls.REQUIRED_FIELDS if record[field] is None
        ]

        if null_fields:
            raise ValueError(
                f"Null values in required fields: {sorted(null_fields)}"
            )
        try:
            normalized = {
                cls.FIELD_MAPPING.get(key, key): value for key, value in record.items()
            }
            normalized["date"] = date.fromisoformat(normalized["date"])
            for field in (
                "open",
                "high",
                "low",
                "close",
                "change",
                "change_percent",
                "vwap",
            ):
                normalized[field] = float(normalized[field])
            normalized["volume"] = int(normalized["volume"])
            if not normalized["symbol"].strip():
                raise ValueError("symbol cannot be empty")

            if normalized["volume"] < 0:
                raise ValueError("volume cannot be negative")

            if normalized["low"] > min(
                normalized["open"],
                normalized["close"],
            ):
                raise ValueError("low must be <= open and close")
            if normalized["high"] < max(
                normalized["open"],
                normalized["close"],
            ):
                raise ValueError("high must be >= open and close")
            return MarketData(**normalized)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid market data record: {exc}") from exc

    @classmethod
    def normalize_many(cls, records: list[dict[str, Any]]) -> list[MarketData]:
        return [cls.normalize(record) for record in records]
