from collector.config import get_settings
from collector.providers.fmp import FMPClient


def main():
    settings = get_settings()
    with FMPClient(settings) as fmp:
        records = fmp.get_historical_prices(
            symbol="^GSPC", start_date="2020-01-01", end_date="2026-11-09"
        )
    print(f"Records received: {len(records)}")
