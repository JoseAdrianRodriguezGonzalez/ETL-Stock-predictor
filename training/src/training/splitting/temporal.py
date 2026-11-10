

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    train_positions: np.ndarray
    validation_positions: np.ndarray
    test_positions: np.ndarray


class TemporalSplitter:
    def __init__(
        self,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        test_size: float = 0.15,
    ) -> None:
        if not np.isclose(
            train_size + validation_size + test_size,
            1.0,
        ):
            raise ValueError(
                "Split sizes must sum to 1.0"
            )

        if any(
            size <= 0
            for size in (
                train_size,
                validation_size,
                test_size,
            )
        ):
            raise ValueError(
                "Split sizes must be greater than zero"
            )

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

    def split(
        self,
        df: pd.DataFrame,
    ) -> TemporalSplit:
        if df.empty:
            raise ValueError("Cannot split an empty dataset")

        required_columns = {"target_date"}

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if df["target_date"].isnull().any():
            raise ValueError("target_date contains null values")

        dates = pd.to_datetime(
            df["target_date"],
            errors="raise",
        )

        if dates.duplicated().any():
            raise ValueError("Duplicate target dates found")

        if not dates.is_monotonic_increasing:
            raise ValueError(
                "Dataset must be sorted by target_date"
            )

        n_samples = len(df)

        train_end = int(n_samples * self.train_size)

        validation_end = train_end + int(
            n_samples * self.validation_size
        )

        if (
            train_end == 0
            or validation_end >= n_samples
        ):
            raise ValueError(
                "Dataset is too small for the requested split"
            )

        train_positions = np.arange(0, train_end)
        validation_positions = np.arange(
            train_end,
            validation_end,
        )
        test_positions = np.arange(
            validation_end,
            n_samples,
        )

        return TemporalSplit(
            train=df.iloc[train_positions].copy(),
            validation=df.iloc[validation_positions].copy(),
            test=df.iloc[test_positions].copy(),
            train_positions=train_positions,
            validation_positions=validation_positions,
            test_positions=test_positions,
        )
