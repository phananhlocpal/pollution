"""Common-field plus zero-mean station-local forecaster."""

from __future__ import annotations

import torch
from torch import nn


def _head(input_dim, hidden_dim, output_dim, dropout):
    module = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )
    nn.init.zeros_(module[-1].weight); nn.init.zeros_(module[-1].bias)
    return module


class CommonLocalForecaster(nn.Module):
    """Forecast station-mean dynamics and zero-mean local dynamics separately."""

    def __init__(self, stations=184, weather_dim=6, horizon=24,
                 hidden_dim=48, horizon_dim=16, station_dim=8, dropout=.1,
                 gru_layers=1):
        super().__init__()
        self.horizon, self.weather_dim = horizon, weather_dim
        gru_dropout = dropout if gru_layers > 1 else 0.0
        self.common_gru = nn.GRU(
            1 + weather_dim + 1, hidden_dim, num_layers=gru_layers,
            dropout=gru_dropout, batch_first=True,
        )
        self.local_gru = nn.GRU(
            3 + weather_dim * 3, hidden_dim, num_layers=gru_layers,
            dropout=gru_dropout, batch_first=True,
        )
        self.horizon_embedding = nn.Embedding(horizon, horizon_dim)
        self.station_embedding = nn.Embedding(stations, station_dim)
        self.common_head = _head(
            hidden_dim + horizon_dim + weather_dim, hidden_dim, 1, dropout
        )
        self.local_head = _head(
            hidden_dim + horizon_dim + station_dim + weather_dim,
            hidden_dim, 1, dropout,
        )

    @staticmethod
    def zero_station_mean(value):
        return value - value.mean(-1, keepdim=True)

    def forward(self, batch):
        x, future_weather = batch["x"], batch["future_weather"]
        pm, weather = x[..., :1], x[..., 1:]
        batch_size, history, stations, _ = x.shape
        common_pm = pm.mean(2)
        residual_pm = pm - common_pm[:, :, None]
        regional_weather = weather.mean(2)
        common_input = torch.cat((
            common_pm, regional_weather,
            x.new_ones(batch_size, history, 1),
        ), -1)
        local_input = torch.cat((
            residual_pm, torch.ones_like(residual_pm), torch.zeros_like(residual_pm),
            weather, torch.ones_like(weather), torch.zeros_like(weather),
        ), -1)
        _, common_state = self.common_gru(common_input)
        flattened = local_input.permute(0, 2, 1, 3).reshape(
            batch_size * stations, history, -1
        )
        _, local_state = self.local_gru(flattened)
        local_state = local_state[-1].reshape(batch_size, stations, -1)
        horizon_embedding = self.horizon_embedding(
            torch.arange(self.horizon, device=x.device)
        )
        regional_future = future_weather.mean(2)
        common_context = common_state[-1, :, None].expand(-1, self.horizon, -1)
        common_delta = self.common_head(torch.cat((
            common_context,
            horizon_embedding[None].expand(batch_size, -1, -1),
            regional_future,
        ), -1)).squeeze(-1)
        station_embedding = self.station_embedding(
            torch.arange(stations, device=x.device)
        )
        local_context = local_state[:, None].expand(-1, self.horizon, -1, -1)
        horizon_nodes = horizon_embedding[None, :, None].expand(
            batch_size, -1, stations, -1
        )
        station_nodes = station_embedding[None, None].expand(
            batch_size, self.horizon, -1, -1
        )
        local_delta = self.local_head(torch.cat((
            local_context, horizon_nodes, station_nodes, future_weather,
        ), -1)).squeeze(-1)
        local_delta = self.zero_station_mean(local_delta)

        last_pm = pm[:, -1, :, 0]
        last_common = last_pm.mean(-1)
        last_residual = last_pm - last_common[:, None]
        common_prediction = last_common[:, None] + common_delta
        residual_prediction = last_residual[:, None] + local_delta
        prediction = common_prediction[:, :, None] + residual_prediction
        return {
            "prediction": prediction,
            "common_prediction": common_prediction,
            "residual_prediction": residual_prediction,
            "persistence": last_pm[:, None].expand_as(prediction),
        }

