from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from collector.normalization.market_data import MarketData


@dataclass(frozen=True)
class UpsertResult:
    modified: int
    matched: int
    upserted: int


class MarketDataRepository:
    COLLECTION_NAME = "historical_prices"

    def __init__(self, database: Database) -> None:
        self.collection: Collection = database[self.COLLECTION_NAME]

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [
                ("symbol", ASCENDING),
                ("date", ASCENDING),
            ],
            unique=True,
            name="unique_symbol_date",
        )

    @staticmethod
    def _to_document(record: MarketData) -> dict[str, Any]:
        document = dict(record)

        document["date"] = datetime.combine(
            record["date"],
            time.min,
            tzinfo=UTC,
        )

        return document

    def upsert_many(self, records: list[MarketData]) -> UpsertResult:
        if not records:
            return UpsertResult(
                modified=0,
                matched=0,
                upserted=0,
            )

        operations = [
            UpdateOne(
                {
                    "symbol": record["symbol"],
                    "date": datetime.combine(
                        record["date"],
                        time.min,
                        tzinfo=UTC,
                    ),
                },
                {"$set": self._to_document(record)},
                upsert=True,
            )
            for record in records
        ]

        result = self.collection.bulk_write(
            operations,
            ordered=False,
        )

        return UpsertResult(
            modified=result.modified_count,
            matched=result.matched_count,
            upserted=result.upserted_count,
        )

    def get_latest_record(
        self,
        symbol: str,
    ) -> dict[str, Any] | None:
        document = self.collection.find_one(
            {"symbol": symbol},
            sort=[("date", -1)],
        )

        if document is None:
            return None

        document["date"] = document["date"].date()

        return document

    def get_latest_date(self, symbol: str) -> date | None:
        record = self.get_latest_record(symbol)

        if record is None:
            return None

        return record["date"]

    def get_records(self, symbol: str) -> list[MarketData]:
        documents = self.collection.find(
            {"symbol": symbol},
            {"_id": 0},
        ).sort("date", ASCENDING)

        records: list[MarketData] = []

        for document in documents:
            document["date"] = document["date"].date()
            records.append(MarketData(**document))

        return records
