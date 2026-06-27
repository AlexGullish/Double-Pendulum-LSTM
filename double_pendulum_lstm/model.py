from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)

    def architecture_summary(self) -> str:
        return (
            f"LSTM(input_size={self.lstm.input_size}, hidden_size={self.lstm.hidden_size}, "
            f"num_layers={self.lstm.num_layers}, dropout={self.dropout.p}) "
            "-> Linear(output_size=4)"
        )
