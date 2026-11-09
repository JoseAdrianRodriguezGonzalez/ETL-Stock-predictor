import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from collector.config import Settings


class FMPError(Exception):
    """Base exception for Financial Modeling Prep errors."""


class FMPAPIError(FMPError):
    """Raised when FMP returns an unsuccessful HTTP response."""


class FMPConnectionError(FMPError):
    """Raised when communication with FMP fails."""


class FMPResponseError(FMPError):
    """Raised when FMP returns an unexpected response."""


class FMPClient:
    ENDPOINT = "historical-price-eod/full"

    def __init__(self,settings: Settings,transport: httpx.BaseTransport | None = None) -> None:
        self.max_retries = settings.fmp_max_retries
        self.client = httpx.Client(base_url=f"{settings.fmp_base_url.rstrip('/')}/",
            headers={"apikey": settings.fmp_api_key},
            timeout=settings.fmp_timeout,
            transport=transport
        )

    def get_historical_prices(self,symbol: str,start_date: str | None = None,end_date: str | None = None) -> list[dict[str, Any]]:
        if not symbol or not symbol.strip():
            raise ValueError("symbol cannot be empty")

        params: dict[str, str] = {"symbol": symbol.strip()}

        if start_date:
            params["from"] = start_date

        if end_date:
            params["to"] = end_date

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(self.ENDPOINT, params=params)

            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise FMPConnectionError(
                        "Could not communicate with FMP"
                    ) from exc

                self._wait(attempt)
                continue

            if (response.status_code in (408, 429) or response.status_code >= 500) and attempt < self.max_retries:
                    self._wait(attempt, response.headers.get("Retry-After"))
                    continue

            if response.is_error:
                raise FMPAPIError(
                    f"FMP returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise FMPResponseError("FMP returned invalid JSON") from exc

            if not isinstance(data, list):
                raise FMPResponseError(
                    f"Expected a list, received {type(data).__name__}"
                )

            if not all(isinstance(row, dict) for row in data):
                raise FMPResponseError(
                    "FMP returned records with an unexpected structure"
                )

            return data

        raise FMPConnectionError("FMP request failed after all retries")

    def _wait(self,attempt: int,retry_after: str | None = None) -> None:
        delay = min(2**attempt, 30)
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    retry_date = parsedate_to_datetime(retry_after)

                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=UTC)

                    delay = (
                        retry_date - datetime.now(UTC)
                    ).total_seconds()

                except (TypeError, ValueError, OverflowError):
                    pass

        time.sleep(min(max(0, delay), 60))

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "FMPClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
