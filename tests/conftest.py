

import pytest


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
