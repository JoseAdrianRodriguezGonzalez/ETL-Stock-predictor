import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MarketDataPreprocessor:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns: list[str] | None = None

    def fit(
        self,
        df: pd.DataFrame,
        train_positions: np.ndarray,
        feature_columns: list[str],
    ) -> None:
        if df.empty:
            raise ValueError("Dataset is empty")

        if not feature_columns:
            raise ValueError("No feature columns provided")

        train_positions = np.asarray(
            train_positions,
            dtype=np.int64,
        )

        if train_positions.ndim != 1 or len(train_positions) == 0:
            raise ValueError(
                "train_positions must be a non-empty 1D array"
            )

        if (
            train_positions.min() < 0
            or train_positions.max() >= len(df)
        ):
            raise ValueError(
                "Training position is outside dataset bounds"
            )

        missing = set(feature_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing feature columns: {sorted(missing)}"
            )

        if len(np.unique(train_positions)) != len(train_positions):
            raise ValueError(
                "Training positions contain duplicates"
            )

        train_features = df.iloc[
            train_positions
        ][feature_columns]

        if train_features.isnull().any().any():
            raise ValueError(
                "Training features contain null values"
            )

        if not np.isfinite(
            train_features.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Training features contain non-finite values"
            )

        self.scaler.fit(train_features)

        self.feature_columns = list(feature_columns)
        self.is_fitted = True

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.feature_columns is None:
            raise RuntimeError(
                "Preprocessor must be fitted before transform"
            )

        missing = (
            set(self.feature_columns) - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing feature columns: {sorted(missing)}"
            )

        features = df[self.feature_columns]

        if features.isnull().any().any():
            raise ValueError(
                "Features contain null values"
            )

        if not np.isfinite(
            features.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Features contain non-finite values"
            )

        transformed = self.scaler.transform(features)

        result = df.copy()

        result.loc[:, self.feature_columns] = transformed

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        train_positions: np.ndarray,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        self.fit(
            df=df,
            train_positions=train_positions,
            feature_columns=feature_columns,
        )

        return self.transform(df)
