"""Minimal frozen-baseline residual corrections motivated by residual EDA."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from benchmarking.residual_probe import neighbor_weights

from .model import CommonLocalForecaster


COMPONENT_DIMS = {"spatial": 2, "wind": 2, "meteo": 2, "regional": 1}


def spatial_buffers(city_path, regions=8):
    city = np.loadtxt(city_path, dtype=str)
    coordinates = city[:, 2:4].astype(float)
    weights = neighbor_weights(coordinates, neighbors=min(8, len(coordinates) - 1)).astype(np.float32)
    lat = np.deg2rad(coordinates[:, 1]); lon = np.deg2rad(coordinates[:, 0])
    north = lat[:, None] - lat[None, :]
    east = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
    norm = np.hypot(east, north)
    east = np.divide(east, norm, out=np.zeros_like(east), where=norm > 0).astype(np.float32)
    north = np.divide(north, norm, out=np.zeros_like(north), where=norm > 0).astype(np.float32)
    # Deterministic geographic quantile grid: 4 longitude bands x 2 latitude bands.
    lon_band = np.digitize(coordinates[:, 0], np.quantile(coordinates[:, 0], (.25, .5, .75)))
    lat_band = np.digitize(coordinates[:, 1], np.quantile(coordinates[:, 1], (.5,)))
    labels = lon_band * 2 + lat_band
    projection = np.zeros_like(weights)
    for label in range(regions):
        mask = labels == label
        projection[np.ix_(mask, mask)] = 1 / max(mask.sum(), 1)
    return weights, east, north, projection.astype(np.float32)


class FrozenResidualCorrection(nn.Module):
    """Add a small zero-mean correction to a frozen common_local prediction."""

    def __init__(self, components, city_path, feature_mean, feature_std, hidden_dim=16):
        super().__init__()
        self.components = tuple(components)
        unknown = set(self.components) - set(COMPONENT_DIMS)
        if unknown:
            raise ValueError(f"Unknown correction components: {sorted(unknown)}")
        self.base = CommonLocalForecaster()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        weights, east, north, projection = spatial_buffers(city_path)
        self.register_buffer("neighbor_weights", torch.from_numpy(weights))
        self.register_buffer("bearing_east", torch.from_numpy(east))
        self.register_buffer("bearing_north", torch.from_numpy(north))
        self.register_buffer("regional_projection", torch.from_numpy(projection))
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.horizon_embedding = nn.Embedding(24, 4)
        signal_dim = sum(COMPONENT_DIMS[name] for name in self.components)
        self.correction_head = nn.Sequential(
            nn.Linear(signal_dim + 4, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def _wind_innovation(self, x, lag):
        pm = x[:, -1 - lag, :, 0]
        sin_from = x[:, -1 - lag, :, 5] * self.feature_std[5] + self.feature_mean[5]
        cos_from = x[:, -1 - lag, :, 6] * self.feature_std[6] + self.feature_mean[6]
        flow_east, flow_north = -sin_from, -cos_from
        alignment = torch.relu(
            self.bearing_east[None] * flow_east[:, None, :]
            + self.bearing_north[None] * flow_north[:, None, :]
        )
        weights = self.neighbor_weights[None] * alignment
        total = weights.sum(-1, keepdim=True)
        normalized = torch.where(
            total > 1e-8, weights / total.clamp_min(1e-8), self.neighbor_weights[None]
        )
        return torch.einsum("bij,bj->bi", normalized, pm) - pm

    def _signals(self, batch):
        x = batch["x"]; current = x[:, -1, :, 0]
        signals = []
        if "spatial" in self.components:
            innovation = current @ self.neighbor_weights.T - current
            neighbor_trend = (current - x[:, -2, :, 0]) @ self.neighbor_weights.T
            signals.extend((innovation[:, None], neighbor_trend[:, None]))
        if "wind" in self.components:
            signals.extend((self._wind_innovation(x, 0)[:, None], self._wind_innovation(x, 1)[:, None]))
        if "meteo" in self.components:
            auxiliary = batch["future_auxiliary"]
            signals.extend((auxiliary[..., 2], auxiliary[..., 3]))
        if "regional" in self.components:
            regional = current @ self.regional_projection.T - current
            signals.append(regional[:, None])
        expanded = []
        for signal in signals:
            if signal.shape[1] == 1:
                signal = signal.expand(-1, 24, -1)
            expanded.append(signal[..., None])
        return torch.cat(expanded, dim=-1)

    def forward(self, batch):
        with torch.no_grad():
            output = self.base(batch)
        signals = self._signals(batch)
        batch_size, horizon, stations, _ = signals.shape
        horizon_embedding = self.horizon_embedding(torch.arange(horizon, device=signals.device))
        horizon_embedding = horizon_embedding[None, :, None].expand(batch_size, -1, stations, -1)
        correction = self.correction_head(torch.cat((signals, horizon_embedding), -1)).squeeze(-1)
        correction = correction - correction.mean(-1, keepdim=True)
        output["prediction"] = output["prediction"] + correction
        output["residual_prediction"] = output["residual_prediction"] + correction
        output["correction"] = correction
        return output
