"""Sequential transport/source operator for multi-step PM2.5 forecasting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .correction import spatial_buffers


def _zero_last(module: nn.Sequential) -> nn.Sequential:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


class TransportSourceRecurrentForecaster(nn.Module):
    """Evolve a PM field one forecast step at a time.

    The transport contribution is conservative across stations.  The
    meteorological source/sink contribution is split into an unconstrained
    common term and a zero-mean station-local term.
    """

    def __init__(
        self,
        city_path,
        stations: int = 184,
        weather_dim: int = 6,
        auxiliary_dim: int = 3,
        horizon: int = 24,
        hidden_dim: int = 64,
        station_dim: int = 8,
        month_dim: int = 4,
        operator_dim: int = 32,
        max_step: float = 0.5,
        event_expert: bool = False,
        use_transport: bool = True,
        use_source: bool = True,
        use_lagged_transport: bool = True,
        use_auxiliary: bool = True,
        use_month: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_step = max_step
        self.event_expert = event_expert
        self.use_transport = use_transport
        self.use_source = use_source
        self.use_lagged_transport = use_lagged_transport
        self.use_auxiliary = use_auxiliary
        self.use_month = use_month
        self.common_encoder = nn.GRU(1 + weather_dim + 1, hidden_dim, batch_first=True)
        self.local_encoder = nn.GRU(3 + weather_dim * 3, hidden_dim, batch_first=True)
        self.station_embedding = nn.Embedding(stations, station_dim)
        self.month_embedding = nn.Embedding(12, month_dim)

        common_future_dim = 1 + weather_dim + auxiliary_dim + month_dim
        local_future_dim = (
            1 + weather_dim + auxiliary_dim + month_dim + station_dim + 2
        )
        self.common_cell = nn.GRUCell(common_future_dim, hidden_dim)
        self.local_cell = nn.GRUCell(local_future_dim, hidden_dim)
        self.transport_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + 3, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        self.common_source_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + common_future_dim, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        self.local_source_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + local_future_dim - 2, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        if event_expert:
            self.event_source_head = _zero_last(nn.Sequential(
                nn.Linear(hidden_dim + local_future_dim - 2, operator_dim), nn.GELU(),
                nn.Linear(operator_dim, 1),
            ))
            self.event_gate = nn.Sequential(
                nn.Linear(hidden_dim + local_future_dim, operator_dim), nn.GELU(),
                nn.Linear(operator_dim, 1),
            )
            nn.init.constant_(self.event_gate[-1].bias, -2.0)

        weights, east, north, _ = spatial_buffers(city_path)
        indices = []
        edge_weights = []
        edge_east = []
        edge_north = []
        for target in range(stations):
            neighbors = np.flatnonzero(weights[target] > 0)
            indices.append(neighbors)
            edge_weights.append(weights[target, neighbors])
            edge_east.append(east[target, neighbors])
            edge_north.append(north[target, neighbors])
        if len({len(row) for row in indices}) != 1:
            raise ValueError("KNN graph must have the same number of edges per station")
        self.register_buffer("neighbor_index", torch.as_tensor(np.stack(indices), dtype=torch.long))
        self.register_buffer("edge_weight", torch.as_tensor(np.stack(edge_weights), dtype=torch.float32))
        self.register_buffer("edge_east", torch.as_tensor(np.stack(edge_east), dtype=torch.float32))
        self.register_buffer("edge_north", torch.as_tensor(np.stack(edge_north), dtype=torch.float32))
        self.register_buffer("station_threshold", torch.zeros(stations, dtype=torch.float32))

    @staticmethod
    def _zero_mean(value: torch.Tensor) -> torch.Tensor:
        return value - value.mean(-1, keepdim=True)

    def _wind_innovation(
        self, state: torch.Tensor, wind_sin: torch.Tensor, wind_cos: torch.Tensor
    ) -> torch.Tensor:
        neighbor_state = state[:, self.neighbor_index]
        flow_east = (-wind_sin)[:, self.neighbor_index]
        flow_north = (-wind_cos)[:, self.neighbor_index]
        alignment = torch.relu(
            self.edge_east[None] * flow_east + self.edge_north[None] * flow_north
        )
        weights = self.edge_weight[None] * alignment
        total = weights.sum(-1, keepdim=True)
        fallback = self.edge_weight[None].expand_as(weights)
        weights = torch.where(total > 1e-8, weights / total.clamp_min(1e-8), fallback)
        return (weights * neighbor_state).sum(-1) - state

    def forward(self, batch):
        x = batch["x"]
        future_weather = batch["future_weather"]
        auxiliary = batch["future_auxiliary"][..., (2, 3, 4)]
        if not self.use_auxiliary:
            auxiliary = torch.zeros_like(auxiliary)
        batch_size, history, stations, _ = x.shape
        pm, weather = x[..., :1], x[..., 1:]
        common_pm = pm.mean(2)
        residual_pm = pm - common_pm[:, :, None]
        common_input = torch.cat((
            common_pm, weather.mean(2), x.new_ones(batch_size, history, 1),
        ), -1)
        local_input = torch.cat((
            residual_pm, torch.ones_like(residual_pm), torch.zeros_like(residual_pm),
            weather, torch.ones_like(weather), torch.zeros_like(weather),
        ), -1)
        _, common_hidden = self.common_encoder(common_input)
        flattened = local_input.permute(0, 2, 1, 3).reshape(batch_size * stations, history, -1)
        _, local_hidden = self.local_encoder(flattened)
        common_hidden = common_hidden[-1]
        local_hidden = local_hidden[-1].reshape(batch_size, stations, -1)

        station = self.station_embedding(torch.arange(stations, device=x.device))
        station = station[None].expand(batch_size, -1, -1)
        state_history = [pm[:, -2, :, 0], pm[:, -1, :, 0]]
        predictions, transports, sources, gates = [], [], [], []
        for step in range(self.horizon):
            state = state_history[-1]
            weather_step = future_weather[:, step]
            auxiliary_step = auxiliary[:, step]
            month = self.month_embedding(batch["future_month"][:, step])
            if not self.use_month:
                month = torch.zeros_like(month)
            month_nodes = month[:, None].expand(-1, stations, -1)
            common_features = torch.cat((
                state.mean(-1, keepdim=True), weather_step.mean(1),
                auxiliary_step.mean(1), month,
            ), -1)
            common_hidden = self.common_cell(common_features, common_hidden)

            innovation_now = self._wind_innovation(
                state, weather_step[..., 4], weather_step[..., 5]
            )
            innovation_lag = self._wind_innovation(
                state_history[-2], weather_step[..., 4], weather_step[..., 5]
            )
            if not self.use_lagged_transport:
                innovation_lag = torch.zeros_like(innovation_lag)
            if not self.use_transport:
                innovation_now = torch.zeros_like(innovation_now)
                innovation_lag = torch.zeros_like(innovation_lag)
            residual_state = state - state.mean(-1, keepdim=True)
            local_features = torch.cat((
                residual_state[..., None], weather_step, auxiliary_step,
                month_nodes, station,
            ), -1)
            local_recurrent = torch.cat((
                local_features, innovation_now[..., None], innovation_lag[..., None],
            ), -1)
            local_hidden = self.local_cell(
                local_recurrent.reshape(batch_size * stations, -1),
                local_hidden.reshape(batch_size * stations, -1),
            ).reshape(batch_size, stations, -1)

            transport_input = torch.cat((
                local_hidden, innovation_now[..., None], innovation_lag[..., None],
                weather_step[..., 3:4],
            ), -1)
            transport = self.max_step * torch.tanh(self.transport_head(transport_input).squeeze(-1))
            transport = self._zero_mean(transport)
            if not self.use_transport:
                transport = torch.zeros_like(transport)
            common_source = self.max_step * torch.tanh(
                self.common_source_head(torch.cat((common_hidden, common_features), -1)).squeeze(-1)
            )
            local_source = self.max_step * torch.tanh(
                self.local_source_head(torch.cat((local_hidden, local_features), -1)).squeeze(-1)
            )
            if self.event_expert:
                neighbor_extreme = (
                    state[:, self.neighbor_index] > self.station_threshold[self.neighbor_index][None]
                ).to(state.dtype).mean(-1)
                relative_level = state - self.station_threshold[None]
                gate_features = torch.cat((
                    local_hidden, local_features, relative_level[..., None],
                    neighbor_extreme[..., None],
                ), -1)
                gate = torch.sigmoid(self.event_gate(gate_features).squeeze(-1))
                event_source = self.max_step * torch.tanh(
                    self.event_source_head(
                        torch.cat((local_hidden, local_features), -1)
                    ).squeeze(-1)
                )
                local_source = (1 - gate) * local_source + gate * event_source
                gates.append(gate)
            local_source = self._zero_mean(local_source)
            source = common_source[:, None] + local_source
            if not self.use_source:
                source = torch.zeros_like(source)
            next_state = state + transport + source
            state_history.append(next_state)
            predictions.append(next_state)
            transports.append(transport)
            sources.append(source)

        prediction = torch.stack(predictions, 1)
        common_prediction = prediction.mean(-1)
        persistence = pm[:, -1, :, 0][:, None].expand_as(prediction)
        result = {
            "prediction": prediction,
            "common_prediction": common_prediction,
            "residual_prediction": prediction - common_prediction[:, :, None],
            "persistence": persistence,
            "transport_operator": torch.stack(transports, 1),
            "source_operator": torch.stack(sources, 1),
        }
        if gates:
            result["event_gate"] = torch.stack(gates, 1)
        return result
