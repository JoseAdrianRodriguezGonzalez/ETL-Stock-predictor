

import os
import joblib
import numpy as np
import pandas as pd

from backend.config import settings
from backend.services.feature_service import FEATURE_COLUMNS


class PreprocessingService:

    def __init__(self):

        if not os.path.isfile(settings.scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {settings.scaler_path}"
            )

        artifact = joblib.load(settings.scaler_path)

        self.scaler = artifact["scaler"]
        self.feature_columns = artifact["feature_columns"]

        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                "Scaler feature columns do not match "
                "the inference configuration."
            )

    def prepare_input(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        features = df[self.feature_columns]

        transformed = self.scaler.transform(features)

        transformed = np.asarray(
            transformed,
            dtype=np.float32,
        )

        sequence_length = settings.sequence_length

        if len(transformed) < sequence_length:
            raise ValueError(
                f"At least {sequence_length} observations "
                "are required."
            )

        window = transformed[-sequence_length:]

        inputs = np.expand_dims(window, axis=0)

        if inputs.shape != (1, sequence_length, len(self.feature_columns)):
            raise ValueError(
                f"Unexpected inference input shape: {inputs.shape}"
            )

        if not np.isfinite(inputs).all():
            raise ValueError(
                "Inference input contains non-finite values."
            )

        return inputs

