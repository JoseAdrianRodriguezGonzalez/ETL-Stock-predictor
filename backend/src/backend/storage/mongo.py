

from pymongo import MongoClient
from pymongo.database import Database

from backend.config import settings


class MongoConnection:
    def __init__(self) -> None:
        self.client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )

        self.database: Database = self.client[
            settings.mongo_database
        ]

    def connect(self) -> None:
        self.client.admin.command("ping")

    def close(self) -> None:
        self.client.close()
