

from dataclasses import dataclass
import os

import pandas as pd

from training.datasets.market_data import MarketDataLoader
from training.features.market_features import (
    FEATURE_COLUMNS,
    MarketFeatureEngineer,
)
from training.preprocessing.market_data import MarketDataPreprocessor
from training.splitting.temporal import (
    TemporalSplit,
    TemporalSplitter,
)
from training.windows.temporal import (
    TemporalWindowGenerator,
    TemporalWindows,
)


@dataclass(frozen=True)
class MarketDataResult:
    train: TemporalWindows
    validation: TemporalWindows
    test: TemporalWindows
    split: TemporalSplit


class MarketDataPipeline:
    def __init__(
        self,
        sequence_length: int = 30,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        test_size: float = 0.15,
    ) -> None:
        self.loader = MarketDataLoader()
        self.feature_engineer = MarketFeatureEngineer()

        self.splitter = TemporalSplitter(
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
        self.preprocessor=MarketDataPreprocessor()
        self.window_generator = TemporalWindowGenerator(
            sequence_length=sequence_length,
        )

    def run(
        self,
        dataset_path: str,
        expected_symbol: str | None = None,
    ) -> MarketDataResult:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        # 1. Load
        df = self.loader.load(
            dataset_path,
            expected_symbol=expected_symbol,
        )

        # 2. Feature engineering
        df = self.feature_engineer.transform(df)

        # 3. Temporal split
        split = self.splitter.split(df)
        df = self.preprocessor.fit_transform(
            df=df,
            train_positions=split.train_positions,
            feature_columns=FEATURE_COLUMNS,
        )
        # 4. Temporal windows
        train = self.window_generator.generate(
            df=df,
            target_positions=split.train_positions,
            feature_columns=FEATURE_COLUMNS,
        )

        validation = self.window_generator.generate(
            df=df,
            target_positions=split.validation_positions,
            feature_columns=FEATURE_COLUMNS,
        )

        test = self.window_generator.generate(
            df=df,
            target_positions=split.test_positions,
            feature_columns=FEATURE_COLUMNS,
        )

        return MarketDataResult(
            train=train,
            validation=validation,
            test=test,
            split=split,
        )

