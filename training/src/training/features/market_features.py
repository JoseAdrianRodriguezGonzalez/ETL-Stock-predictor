import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "daily_return",
    "overnight_gap",
    "intraday_return",
    "high_low_range",
    "log_volume",
    "vwap_deviation",
]


class MarketFeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        previous_close = df["close"].shift(1)

        df["daily_return"] = (
            df["close"] / previous_close
        ) - 1

        df["overnight_gap"] = (
            df["open"] / previous_close
        ) - 1

        df["intraday_return"] = (
            df["close"] / df["open"]
        ) - 1

        df["high_low_range"] = (
            (df["high"] - df["low"]) / previous_close
        )

        df["log_volume"] = np.log1p(df["volume"])

        df["vwap_deviation"] = (
            df["vwap"] / df["close"]
        ) - 1

        # Target: rendimiento de la siguiente sesión.
        df["target"] = (
            df["close"].shift(-1) / df["close"]
        ) - 1

        # Fecha correspondiente al objetivo.
        df["target_date"] = df["date"].shift(-1)

        df = df.dropna(
            subset=FEATURE_COLUMNS + ["target", "target_date"]
        )

        return df.reset_index(drop=True)
