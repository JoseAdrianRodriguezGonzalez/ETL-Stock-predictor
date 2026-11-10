from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "change",
    "change_percent",
    "vwap",
}

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "change",
    "change_percent",
    "vwap",
]


class MarketDataLoader:
    def load(
        self,
        path: str | Path,
        expected_symbol: str | None = None,
    ) -> pd.DataFrame:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Market data file not found: {path}"
            )

        df = pd.read_parquet(path)

        missing_columns = REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        df = df[list(REQUIRED_COLUMNS)].copy()

        if df.empty:
            raise ValueError("Market data is empty")

        if df.isnull().any().any():
            raise ValueError("Market data contains null values")

        df["date"] = pd.to_datetime(
            df["date"],
            errors="raise",
        )

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        if not df["symbol"].str.strip().ne("").all():
            raise ValueError("Market data contains empty symbols")

        symbols = df["symbol"].unique()

        if len(symbols) != 1:
            raise ValueError(
                "Expected market data for exactly one symbol"
            )

        if expected_symbol and symbols[0] != expected_symbol:
            raise ValueError(
                f"Expected symbol {expected_symbol}, "
                f"received {symbols[0]}"
            )

        if df.duplicated(subset=["symbol", "date"]).any():
            raise ValueError("Market data contains duplicate dates")

        df = df.sort_values("date").reset_index(drop=True)

        return df
