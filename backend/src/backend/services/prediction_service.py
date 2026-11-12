

import numpy as np
import pandas as pd
from pymongo.database import Database

from backend.config import settings
from backend.clients.triton_client import TritonClient
from backend.repositories.market_data_repository import MarketDataRepository
from backend.services.feature_service import FeatureService
from backend.services.preprocessing_service import PreprocessingService


class PredictionService:
    def __init__(self, database: Database) -> None:
        self.repository = MarketDataRepository(database)
        self.feature_service = FeatureService()
        self.preprocessing_service = PreprocessingService()
        self.triton_client = TritonClient()

    def predict_next_day(self, symbol: str) -> dict:
        historical_data = self.repository.get_historical_data(
            symbol=symbol,
        )

        features = self.feature_service.transform(historical_data)

        inputs = self.preprocessing_service.prepare_input(features)

        output = self.triton_client.predict(inputs)
        output = np.asarray(output, dtype=np.float32)

        if output.shape != (1, 1):
            raise ValueError(
                f"Unexpected Triton output shape: {output.shape}"
            )

        predicted_return = float(output[0, 0])

        if not np.isfinite(predicted_return):
            raise ValueError("Model returned a non-finite prediction.")

        last_date = pd.to_datetime(
            historical_data["date"].iloc[-1]
        )

        return {
            "symbol": symbol,
            "last_observed_date": last_date.date().isoformat(),
            "forecast_horizon": "next_trading_session",
            "predicted_return": predicted_return,
            "predicted_return_percent": predicted_return * 100.0,
            "input_shape": list(inputs.shape),
        }

