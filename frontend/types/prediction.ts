

export interface PredictionRequest {
  symbol: string;
}

export interface PredictionResponse {
  symbol: string;
  last_observed_date: string;
  forecast_horizon: "next_trading_session";
  predicted_return: number;
  predicted_return_percent: number;
  input_shape: number[];
}
