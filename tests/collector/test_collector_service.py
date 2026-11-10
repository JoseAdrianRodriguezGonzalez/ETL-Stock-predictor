from datetime import date
from unittest.mock import MagicMock

import pytest

from collector.normalization.market_data import MarketDataNormalizer
from collector.providers.fmp import FMPClient
from collector.repositories.market_data import (
    MarketDataRepository,
    UpsertResult,
)
from collector.services.collector import CollectorService


@pytest.fixture
def fmp_client() -> MagicMock:
    return MagicMock(spec=FMPClient)


@pytest.fixture
def repository() -> MagicMock:
    return MagicMock(spec=MarketDataRepository)


@pytest.fixture
def collector(
    fmp_client: MagicMock,
    repository: MagicMock,
) -> CollectorService:
    return CollectorService(
        fmp_client=fmp_client,
        repository=repository,
    )


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


def test_collect_historical_success(
    collector: CollectorService,
    fmp_client: MagicMock,
    repository: MagicMock,
    fmp_record: dict,
) -> None:
    expected_result = UpsertResult(
        modified=0,
        matched=0,
        upserted=1,
    )

    fmp_client.get_historical_prices.return_value = [fmp_record]
    repository.upsert_many.return_value = expected_result

    result = collector.collect_historical(
        symbol="^GSPC",
        start_date="2020-01-01",
        end_date="2026-11-09",
    )

    fmp_client.get_historical_prices.assert_called_once_with(
        symbol="^GSPC",
        start_date="2020-01-01",
        end_date="2026-11-09",
    )

    normalized = MarketDataNormalizer.normalize_many([fmp_record])

    repository.upsert_many.assert_called_once_with(normalized)

    assert result == expected_result


def test_collect_historical_empty_records(
    collector: CollectorService,
    fmp_client: MagicMock,
    repository: MagicMock,
) -> None:
    expected_result = UpsertResult(
        modified=0,
        matched=0,
        upserted=0,
    )

    fmp_client.get_historical_prices.return_value = []
    repository.upsert_many.return_value = expected_result

    result = collector.collect_historical(
        symbol="^GSPC",
        start_date="2020-01-01",
        end_date="2026-11-09",
    )

    repository.upsert_many.assert_called_once_with([])

    assert result == expected_result


def test_collect_incremental_success(
    collector: CollectorService,
    fmp_client: MagicMock,
    repository: MagicMock,
    fmp_record: dict,
) -> None:
    repository.get_latest_date.return_value = date(2026, 11, 9)

    expected_result = UpsertResult(
        modified=1,
        matched=1,
        upserted=0,
    )

    fmp_client.get_historical_prices.return_value = [fmp_record]
    repository.upsert_many.return_value = expected_result

    result = collector.collect_incremental(
        symbol="^GSPC",
        overlap_days=3,
    )

    fmp_client.get_historical_prices.assert_called_once_with(
        symbol="^GSPC",
        start_date="2026-11-06",
        end_date=None,
    )

    assert result == expected_result


def test_collect_incremental_without_history(
    collector: CollectorService,
    fmp_client: MagicMock,
    repository: MagicMock,
) -> None:
    repository.get_latest_date.return_value = None

    with pytest.raises(
        ValueError,
        match="No historical data found",
    ):
        collector.collect_incremental("^GSPC")

    fmp_client.get_historical_prices.assert_not_called()
    repository.upsert_many.assert_not_called()
