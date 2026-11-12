

from datetime import datetime
from typing import Any

import pandas as pd
from pymongo import DESCENDING
from pymongo.database import Database

from backend.config import settings


class MarketDataRepository:
    COLLECTION_NAME = "historical_prices"

    REQUIRED_COLUMNS = {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "change_percent",
        "vwap",
    }

    NUMERIC_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "change_percent",
        "vwap",
    ]

    def __init__(self, database: Database) -> None:
        self.collection = database[self.COLLECTION_NAME]

    def get_historical_data(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if limit is None:
            limit = settings.sequence_length + 1

        if limit < settings.sequence_length + 1:
            raise ValueError(
                "At least sequence_length + 1 records are required."
            )

        cursor = (
            self.collection.find(
                {"symbol": symbol},
                {"_id": 0},
            )
            .sort("date", DESCENDING)
            .limit(limit)
        )

        documents: list[dict[str, Any]] = list(cursor)

        if len(documents) < limit:
            raise ValueError(
                f"Insufficient historical data for {symbol}. "
                f"Expected at least {limit} records, "
                f"received {len(documents)}."
            )

        # MongoDB returns descending dates; the feature
        # calculations require chronological order.
        documents.reverse()

        df = pd.DataFrame(documents)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"MongoDB documents are missing columns: {sorted(missing)}"
            )

        df = df[list(self.REQUIRED_COLUMNS)].copy()

        # MongoDB stores dates as BSON datetime values.
        df["date"] = pd.to_datetime(df["date"], errors="raise", utc=True)

        for column in self.NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        if df.isnull().any().any():
            raise ValueError(
                "Historical data contains null values."
            )

        if not df["symbol"].eq(symbol).all():
            raise ValueError(
                f"Unexpected symbol in historical data: {symbol}"
            )

        if df.duplicated(subset=["symbol", "date"]).any():
            raise ValueError(
                "Duplicate symbol/date records were found."
            )

        if not df["date"].is_monotonic_increasing:
            raise ValueError(
                "Historical records are not chronologically ordered."
            )

        return df.reset_index(drop=True)

