

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalWindows:
    X: np.ndarray
    y: np.ndarray
    target_dates: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.X)






class TemporalWindowGenerator:
    def __init__(
        self,
        sequence_length: int = 30,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        self.sequence_length = sequence_length

    def generate(
        self,
        df: pd.DataFrame,
        target_positions: np.ndarray,
        feature_columns: list[str],
        target_column: str = "target",
    ) -> TemporalWindows:
        if df.empty:
            raise ValueError("Dataset is empty")

        required_columns = set(feature_columns)
        required_columns.update(
            {target_column, "target_date"}
        )

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        target_positions = np.asarray(
            target_positions,
            dtype=np.int64,
        )

        if target_positions.ndim != 1:
            raise ValueError(
                "target_positions must be one-dimensional"
            )

        if len(target_positions) == 0:
            raise ValueError(
                "No target positions provided"
            )

        if (
            target_positions.min() < 0
            or target_positions.max() >= len(df)
        ):
            raise ValueError(
                "Target position is outside dataset bounds"
            )

        if np.any(np.diff(target_positions) <= 0):
            raise ValueError(
                "Target positions must be unique and ordered"
            )

        if df[feature_columns].isnull().any().any():
            raise ValueError(
                "Feature columns contain null values"
            )

        if df[target_column].isnull().any():
            raise ValueError(
                f"{target_column} contains null values"
            )

        features = df[feature_columns].to_numpy(
            dtype=np.float32,
        )

        targets = df[target_column].to_numpy(
            dtype=np.float32,
        )

        target_dates = pd.to_datetime(
            df["target_date"]
        ).to_numpy()

        X = []
        y = []
        dates = []

        for position in target_positions:
            start = position - self.sequence_length + 1

            if start < 0:
                continue

            window = features[start : position + 1]

            if len(window) != self.sequence_length:
                raise ValueError(
                    "Generated window has an invalid length"
                )

            X.append(window)
            y.append(targets[position])
            dates.append(target_dates[position])

        if not X:
            raise ValueError(
                "No valid temporal windows could be generated"
            )

        return TemporalWindows(
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y, dtype=np.float32),
            target_dates=np.asarray(dates),
        )
