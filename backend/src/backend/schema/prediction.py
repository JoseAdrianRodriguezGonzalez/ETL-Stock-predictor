

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    symbol: str = Field(
        min_length=1,
        max_length=32,
        description="Market symbol to forecast.",
    )


class PredictionResponse(BaseModel):
    symbol: str
    last_observed_date: date
    forecast_horizon: Literal["next_trading_session"]
    predicted_return: float
    predicted_return_percent: float
    input_shape: list[int]
