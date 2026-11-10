import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from collector.normalization.market_data import MarketData


@dataclass(frozen=True)
class DatasetPeriod:
    start: date
    end: date

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    source: str
    symbol: str
    collected_at: datetime
    period: DatasetPeriod
    frequency: str
    records: int
    format: str
    schema_version: str

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "symbol": self.symbol,
            "collected_at": (
                self.collected_at.astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "period": self.period.to_dict(),
            "frequency": self.frequency,
            "records": self.records,
            "format": self.format,
            "schema_version": self.schema_version,
        }


class ManifestGenerator:
    @staticmethod
    def generate(
        records: list[MarketData],
        *,
        dataset: str,
        source: str,
        frequency: str = "1d",
        file_format: str = "parquet",
        schema_version: str = "1.0.0",
        collected_at: datetime | None = None,
    ) -> DatasetManifest:
        if not records:
            raise ValueError("Cannot generate a manifest for an empty dataset")

        symbols = {record["symbol"] for record in records}

        if len(symbols) != 1:
            raise ValueError(
                "All records must belong to the same symbol"
            )

        if not dataset.strip():
            raise ValueError("dataset cannot be empty")

        if not source.strip():
            raise ValueError("source cannot be empty")

        dates = [record["date"] for record in records]

        timestamp = collected_at or datetime.now(UTC)

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("collected_at must be timezone-aware")

        return DatasetManifest(
            dataset=dataset,
            source=source,
            symbol=next(iter(symbols)),
            collected_at=timestamp,
            period=DatasetPeriod(
                start=min(dates),
                end=max(dates),
            ),
            frequency=frequency,
            records=len(records),
            format=file_format,
            schema_version=schema_version,
        )


class ManifestWriter:
    @staticmethod
    def write(
        manifest: DatasetManifest,
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(
            manifest.to_dict(),
            indent=4,
            ensure_ascii=False,
        )

        path.write_text(content + "\n", encoding="utf-8")
