import httpx
import pytest

from collector.config import Settings
from collector.providers.fmp import FMPClient, FMPResponseError


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        log_level="DEBUG",
        fmp_api_key="test-api-key",
        fmp_base_url="https://financialmodelingprep.com/stable",
        fmp_timeout=10,
        fmp_max_retries=3,
        mongo_uri=None,
        mongo_user="test",
        mongo_password="test",
        mongo_database="test_db",
        mongo_host="localhost",
        mongo_port=27017,
    )


def test_get_historical_prices_success(settings):
    expected = [
        {
            "symbol": "^GSPC",
            "date": "2026-11-09",
            "open": 6800.0,
            "high": 6850.0,
            "low": 6780.0,
            "close": 6830.0,
            "volume": 1000000,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=expected)

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        records = client.get_historical_prices(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2026-11-09",
        )

    assert records == expected
    assert len(records) == 1


def test_request_contains_correct_parameters(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/stable/historical-price-eod/full"
        assert request.url.params["symbol"] == "^GSPC"
        assert request.url.params["from"] == "2020-01-01"
        assert request.url.params["to"] == "2026-11-09"

        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        client.get_historical_prices(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2026-11-09",
        )


def test_request_contains_api_key_header(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "test-api-key"
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        client.get_historical_prices("^GSPC")


@pytest.mark.parametrize("symbol", ["", " ", "   "])
def test_empty_symbol_raises_value_error(settings, symbol):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[])
    )

    with FMPClient(settings, transport=transport) as client:
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            client.get_historical_prices(symbol)


def test_invalid_json_raises_response_error(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"invalid json")

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        with pytest.raises(FMPResponseError, match="invalid JSON"):
            client.get_historical_prices("^GSPC")


def test_non_list_response_raises_response_error(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Unexpected response"})

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        with pytest.raises(FMPResponseError, match="Expected a list"):
            client.get_historical_prices("^GSPC")


def test_invalid_record_structure_raises_response_error(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"date": "2026-11-09"}, "invalid"])

    transport = httpx.MockTransport(handler)

    with FMPClient(settings, transport=transport) as client:
        with pytest.raises(
            FMPResponseError,
            match="unexpected structure",
        ):
            client.get_historical_prices("^GSPC")
