

import os

import numpy as np

from training.datasets.market_data import MarketDataLoader
from training.features.market_features import (
    FEATURE_COLUMNS,
    MarketFeatureEngineer,
)
from training.preprocessing.market_data import (
    MarketDataPreprocessor,
)

from backend.services.feature_service import FeatureService


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sp500",
    "historical.parquet",
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "deployment",
    "assets",
    "scaler.joblib",
)


def test_feature_engineering_parity():

    # Load the original historical dataset.
    df = MarketDataLoader().load(DATASET_PATH)

    # Original training implementation.
    training_features = MarketFeatureEngineer().transform(df)

    # Backend implementation.
    backend_features = FeatureService().transform(df)

    # Training removes the last observation because it cannot
    # generate a next-day target for that row.
    # Backend must preserve that observation for inference.
    assert len(backend_features) == len(training_features) + 1

    np.testing.assert_allclose(
        training_features[FEATURE_COLUMNS].to_numpy(),
        backend_features.iloc[:-1][FEATURE_COLUMNS].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_scaler_artifact_compatibility():

    df = MarketDataLoader().load(DATASET_PATH)

    backend_features = FeatureService().transform(df)

    preprocessor = MarketDataPreprocessor.load(
        SCALER_PATH
    )

    assert preprocessor.feature_columns == FEATURE_COLUMNS

    transformed = preprocessor.transform(
        backend_features
    )

    assert transformed[FEATURE_COLUMNS].shape == (
        len(backend_features),
        len(FEATURE_COLUMNS),
    )

    assert np.isfinite(
        transformed[FEATURE_COLUMNS].to_numpy()
    ).all()
