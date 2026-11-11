

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)

        last_hidden = output[:, -1, :]

        prediction = self.output(last_hidden)

        return prediction.squeeze(-1)

