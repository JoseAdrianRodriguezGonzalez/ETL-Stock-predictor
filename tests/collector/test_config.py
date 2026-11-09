import pytest
from dataclasses import FrozenInstanceError
from collector.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    variables = [
        "APP_ENV",
        "LOG_LEVEL",
        "FMP_API_KEY",
        "FMP_BASE_URL",
        "FMP_TIMEOUT",
        "FMP_MAX_RETRIES",
        "MONGO_URI",
        "MONGO_USER",
        "MONGO_PASSWORD",
        "MONGO_DATABASE",
        "MONGO_HOST",
        "MONGO_PORT",
    ]

    for variable in variables:
        monkeypatch.delenv(variable, raising=False)

    monkeypatch.setenv("FMP_API_KEY", "test-api-key")


def test_default_settings(clean_env):
    settings = get_settings()

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"

    assert settings.fmp_api_key == "test-api-key"
    assert settings.fmp_base_url == (
        "https://financialmodelingprep.com/stable"
    )
    assert settings.fmp_timeout == 30.0
    assert settings.fmp_max_retries == 3

    assert settings.mongo_user == "sp500"
    assert settings.mongo_database == "market_data"
    assert settings.mongo_host == "localhost"
    assert settings.mongo_port == 27017


def test_custom_environment_variables(clean_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FMP_TIMEOUT", "60")
    monkeypatch.setenv("FMP_MAX_RETRIES", "5")
    monkeypatch.setenv("MONGO_HOST", "mongodb")
    monkeypatch.setenv("MONGO_PORT", "27018")

    settings = get_settings()

    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.fmp_timeout == 60.0
    assert settings.fmp_max_retries == 5
    assert settings.mongo_host == "mongodb"
    assert settings.mongo_port == 27018


def test_missing_fmp_api_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FMP_API_KEY is required"):
        get_settings()


def test_invalid_timeout(clean_env, monkeypatch):
    monkeypatch.setenv("FMP_TIMEOUT", "0")

    with pytest.raises(ValueError, match="FMP_TIMEOUT"):
        get_settings()


def test_invalid_retries(clean_env, monkeypatch):
    monkeypatch.setenv("FMP_MAX_RETRIES", "-1")

    with pytest.raises(ValueError, match="FMP_MAX_RETRIES"):
        get_settings()


def test_invalid_mongo_port(clean_env, monkeypatch):
    monkeypatch.setenv("MONGO_PORT", "70000")

    with pytest.raises(ValueError, match="MONGO_PORT"):
        get_settings()


def test_mongodb_uri(clean_env, monkeypatch):
    monkeypatch.setenv("MONGO_USER", "sp500")
    monkeypatch.setenv("MONGO_PASSWORD", "password123")
    monkeypatch.setenv("MONGO_HOST", "localhost")
    monkeypatch.setenv("MONGO_PORT", "27017")
    monkeypatch.setenv("MONGO_DATABASE", "market_data")

    settings = get_settings()

    assert settings.mongodb_uri == (
        "mongodb://sp500:password123@localhost:27017/"
        "market_data?authSource=admin"
    )


def test_mongodb_uri_encodes_credentials(clean_env, monkeypatch):
    monkeypatch.setenv("MONGO_USER", "user@admin")
    monkeypatch.setenv("MONGO_PASSWORD", "pass@word/2026")

    settings = get_settings()

    assert settings.mongodb_uri == (
        "mongodb://user%40admin:pass%40word%2F2026"
        "@localhost:27017/market_data?authSource=admin"
    )


def test_mongodb_uri_direct(clean_env, monkeypatch):
    custom_uri = "mongodb://custom-host:27017/market_data"
    monkeypatch.setenv("MONGO_URI", custom_uri)

    settings = get_settings()

    assert settings.mongodb_uri == custom_uri


def test_mongodb_requires_password(clean_env):
    settings = get_settings()

    with pytest.raises(ValueError, match="contraseña"):
        _ = settings.mongodb_uri


def test_settings_are_immutable(clean_env):
    settings = get_settings()

    with pytest.raises(FrozenInstanceError):
        settings.app_env = "production"
