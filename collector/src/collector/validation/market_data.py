import math

from collector.normalization.market_data import MarketData


class MarketDataValidationError(ValueError):
    """Raised when a market data batch violates data quality rules."""


class MarketDataValidator:
    NUMERIC_FIELDS = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "change_percent",
        "vwap",
    )

    @classmethod
    def validate(
        cls,
        records: list[MarketData],
        expected_symbol: str | None = None,
    ) -> None:
        if not records:
            return

        seen: set[tuple[str, object]] = set()
        symbols: set[str] = set()
        previous_date = None
        for index, record in enumerate(records):
            symbol = record["symbol"]
            record_date = record["date"]

            symbols.add(symbol)

            if expected_symbol is not None and symbol != expected_symbol:
                raise MarketDataValidationError(
                    f"Unexpected symbol at record {index}: "
                    f"expected {expected_symbol}, received {symbol}"
                )

            key = (symbol, record_date)

            if key in seen:
                raise MarketDataValidationError(
                    f"Duplicate market data record: {symbol} at {record_date}"
                )
            seen.add(key)
            if previous_date is not None and record_date < previous_date:
                raise MarketDataValidationError(
                    f"Records are not in chronological order at record {index}"
                )

            previous_date = record_date
            for field in cls.NUMERIC_FIELDS:
                value = record[field]

                if not math.isfinite(value):
                    raise MarketDataValidationError(
                        f"Non-finite value in field '{field}' "
                        f"at record {index}"
                    )

            if record["volume"] < 0:
                raise MarketDataValidationError(
                    f"Negative volume at record {index}"
                )

            if record["low"] > record["high"]:
                raise MarketDataValidationError(
                    f"Invalid price range at record {index}: "
                    "low must be <= high"
                )

        if len(symbols) > 1:
            raise MarketDataValidationError(
                f"Multiple symbols found in dataset: {sorted(symbols)}"
            )

