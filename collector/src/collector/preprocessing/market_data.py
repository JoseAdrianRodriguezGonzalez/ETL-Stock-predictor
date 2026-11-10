from collector.normalization.market_data import MarketData


class MarketDataPreprocessor:
    @staticmethod
    def sort_by_date(records: list[MarketData]) -> list[MarketData]:
        return sorted(records, key=lambda record: record["date"])
