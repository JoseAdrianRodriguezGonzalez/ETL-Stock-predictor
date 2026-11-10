from typing import Self

from pymongo import MongoClient
from pymongo.database import Database

from collector.config import Settings


class MongoConnection:
    def __init__(self, settings: Settings) -> None:
        self.client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )

        self.database: Database = self.client[settings.mongo_database]

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        self.ping()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
