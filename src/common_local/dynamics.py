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
        city_path=None,
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
        future_weather_mode: str = "observed",
        weather_hidden_dim: int = 16,
        weather_loss_weight: float = 0.1,
        weather_increment_loss_weight: float = 0.0,
        seasonal_period: int = 8,
        seasonal_weights=None,
        transport_forcing_dim: int = 16,
        source_forcing_dim: int = 32,
        horizon_embedding_dim: int = 8,
        coordinates=None,
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
        if future_weather_mode not in {
            "observed", "persistence", "learned", "latent",
            "seasonal", "seasonal_weighted", "factorized",
        }:
            raise ValueError(f"Unknown future weather mode: {future_weather_mode}")
        self.future_weather_mode = future_weather_mode
        self.latent_forcing = future_weather_mode == "latent"
        self.factorized_forcing = future_weather_mode == "factorized"
        # History-only V2/V3 deliberately remove the explicit lag pathway.
        if self.latent_forcing or self.factorized_forcing:
            self.use_lagged_transport = False
        self.weather_loss_weight = weather_loss_weight
        self.weather_increment_loss_weight = weather_increment_loss_weight
        self.seasonal_period = seasonal_period
        if future_weather_mode == "seasonal_weighted":
            if seasonal_weights is None:
                raise ValueError("seasonal_weighted mode requires train-fitted weights")
            weight_tensor = torch.as_tensor(seasonal_weights, dtype=torch.float32)
            if weight_tensor.ndim != 2 or weight_tensor.shape[0] != weather_dim:
                raise ValueError("seasonal_weights must have shape [weather_dim, cycles]")
            if torch.any(weight_tensor < 0) or not torch.allclose(
                weight_tensor.sum(-1), torch.ones(weather_dim), atol=1e-5
            ):
                raise ValueError("seasonal weights must be non-negative and sum to one")
            self.register_buffer("seasonal_weights", weight_tensor)
        self.common_encoder = nn.GRU(1 + weather_dim + 1, hidden_dim, batch_first=True)
        self.local_encoder = nn.GRU(3 + weather_dim * 3, hidden_dim, batch_first=True)
        if future_weather_mode == "learned":
            self.weather_encoder = nn.GRU(weather_dim, weather_hidden_dim, batch_first=True)
            self.weather_cell = nn.GRUCell(weather_dim, weather_hidden_dim)
            self.weather_head = nn.Linear(weather_hidden_dim, weather_dim)
            nn.init.zeros_(self.weather_head.weight)
            nn.init.zeros_(self.weather_head.bias)
        if self.latent_forcing:
            self.global_forcing_encoder = nn.GRU(
                weather_dim, weather_hidden_dim, batch_first=True
            )
            self.local_forcing_encoder = nn.GRU(
                weather_dim, weather_hidden_dim, batch_first=True
            )
            self.global_forcing_cell = nn.GRUCell(1, weather_hidden_dim)
            self.local_forcing_cell = nn.GRUCell(
                1 + weather_hidden_dim, weather_hidden_dim
            )
            self.latent_transport_signal = nn.Linear(weather_hidden_dim, 1)
        if self.factorized_forcing:
            self.multiscale_lags = (1, 2, 4, 8, 16, 24)
            multiscale_dim = weather_dim * len(self.multiscale_lags)
            self.transport_forcing_encoder = nn.Sequential(
                nn.Linear(multiscale_dim, transport_forcing_dim * 2), nn.GELU(),
                nn.Linear(transport_forcing_dim * 2, transport_forcing_dim),
            )
            self.source_forcing_encoder = nn.Sequential(
                nn.Linear(multiscale_dim, source_forcing_dim * 2), nn.GELU(),
                nn.Linear(source_forcing_dim * 2, source_forcing_dim),
            )
            self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)
            self.transport_forcing_cell = nn.GRUCell(
                horizon_embedding_dim, transport_forcing_dim
            )
            self.source_forcing_cell = nn.GRUCell(
                horizon_embedding_dim, source_forcing_dim
            )
            source_weather_dim = weather_dim - 3 if weather_dim >= 6 else weather_dim
            self.source_weather_head = nn.Linear(source_forcing_dim, source_weather_dim)
            self.transport_weather_head = (
                nn.Linear(transport_forcing_dim, 3) if weather_dim >= 6 else None
            )
            nn.init.zeros_(self.source_weather_head.weight)
            nn.init.zeros_(self.source_weather_head.bias)
            if self.transport_weather_head is not None:
                nn.init.zeros_(self.transport_weather_head.weight)
                nn.init.zeros_(self.transport_weather_head.bias)
        self.station_embedding = nn.Embedding(stations, station_dim)
        self.month_embedding = nn.Embedding(12, month_dim)

        forcing_dim = (
            source_forcing_dim if self.factorized_forcing else
            weather_hidden_dim if self.latent_forcing else weather_dim
        )
        common_future_dim = 1 + forcing_dim + auxiliary_dim + month_dim
        local_feature_dim = 1 + forcing_dim + auxiliary_dim + month_dim + station_dim
        operator_terms = 1 if (self.latent_forcing or self.factorized_forcing) else 2
        local_future_dim = local_feature_dim + operator_terms
        self.common_cell = nn.GRUCell(common_future_dim, hidden_dim)
        self.local_cell = nn.GRUCell(local_future_dim, hidden_dim)
        self.transport_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + operator_terms + 1, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        self.common_source_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + common_future_dim, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        self.local_source_head = _zero_last(nn.Sequential(
            nn.Linear(hidden_dim + local_feature_dim, operator_dim), nn.GELU(),
            nn.Linear(operator_dim, 1),
        ))
        if event_expert:
            self.event_source_head = _zero_last(nn.Sequential(
                nn.Linear(hidden_dim + local_feature_dim, operator_dim), nn.GELU(),
                nn.Linear(operator_dim, 1),
            ))
            self.event_gate = nn.Sequential(
                nn.Linear(hidden_dim + local_feature_dim + 2, operator_dim), nn.GELU(),
                nn.Linear(operator_dim, 1),
            )
            nn.init.constant_(self.event_gate[-1].bias, -2.0)

        weights, east, north, _ = spatial_buffers(city_path, coordinates=coordinates)
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

    def _spatial_innovation(self, state: torch.Tensor) -> torch.Tensor:
        """Current-state graph transport used by history-only latent V2."""
        neighbor_state = state[:, self.neighbor_index]
        weights = self.edge_weight[None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (weights * neighbor_state).sum(-1) - state

    def _initial_latent_forcing(self, weather: torch.Tensor):
        batch_size, history, stations, weather_dim = weather.shape
        global_history = weather.mean(2)
        local_history = weather - global_history[:, :, None]
        _, global_hidden = self.global_forcing_encoder(global_history)
        flattened = local_history.permute(0, 2, 1, 3).reshape(
            batch_size * stations, history, weather_dim
        )
        _, local_hidden = self.local_forcing_encoder(flattened)
        return (
            global_hidden[-1],
            local_hidden[-1].reshape(batch_size, stations, -1),
        )

    def _initial_factorized_forcing(self, weather: torch.Tensor):
        """Encode fixed multi-scale meteorology without consulting PM state."""
        if weather.shape[1] < max(self.multiscale_lags):
            raise ValueError("Weather history is shorter than V3 multi-scale lags")
        selected = torch.stack(
            [weather[:, -lag] for lag in self.multiscale_lags], dim=2
        ).flatten(2)
        return (
            self.transport_forcing_encoder(selected),
            self.source_forcing_encoder(selected),
        )

    def _factorized_weather_step(
        self, previous: torch.Tensor, transport_forcing: torch.Tensor,
        source_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Decode auxiliary meteorology with wind owned by transport state."""
        source_delta = self.source_weather_head(source_forcing)
        if self.transport_weather_head is None:
            return previous + source_delta
        transport_delta = self.transport_weather_head(transport_forcing)
        delta = torch.cat(
            (source_delta[..., :3], transport_delta, source_delta[..., 3:]), -1
        )
        return previous + delta

    def _causal_future_weather(self, weather: torch.Tensor) -> torch.Tensor:
        if self.future_weather_mode == "observed":
            raise RuntimeError("Observed future weather is supplied directly")
        if self.future_weather_mode == "persistence":
            return weather[:, -1:, :, :].expand(-1, self.horizon, -1, -1)
        if self.future_weather_mode in {"seasonal", "seasonal_weighted"}:
            period = self.seasonal_period
            if weather.shape[1] < period:
                raise ValueError("Weather history is shorter than seasonal period")
            if self.future_weather_mode == "seasonal":
                template = weather[:, -period:]
            else:
                cycles = self.seasonal_weights.shape[1]
                if weather.shape[1] < period * cycles:
                    raise ValueError("Weather history is shorter than weighted seasonal cycles")
                histories = torch.stack([
                    weather[:, -(lag + 1) * period:-lag * period if lag else None]
                    for lag in range(cycles)
                ], dim=-1)
                template = (histories * self.seasonal_weights[None, None, None]).sum(-1)
            repeats = (self.horizon + period - 1) // period
            return template.repeat(1, repeats, 1, 1)[:, :self.horizon]
        batch_size, history, stations, weather_dim = weather.shape
        flattened = weather.permute(0, 2, 1, 3).reshape(
            batch_size * stations, history, weather_dim
        )
        _, hidden = self.weather_encoder(flattened)
        hidden = hidden[-1]
        previous = flattened[:, -1]
        predictions = []
        for _ in range(self.horizon):
            hidden = self.weather_cell(previous, hidden)
            # Residual prediction makes persistence the natural initialization.
            previous = previous + self.weather_head(hidden)
            predictions.append(previous.reshape(batch_size, stations, weather_dim))
        return torch.stack(predictions, 1)

    def forward(self, batch):
        x = batch["x"]
        future_weather = batch["future_weather"]
        batch_size, history, stations, _ = x.shape
        if self.use_auxiliary:
            auxiliary = batch["future_auxiliary"][..., (2, 3, 4)]
        else:
            auxiliary = x.new_zeros(batch_size, self.horizon, stations, 3)
        pm, weather = x[..., :1], x[..., 1:]
        causal_weather_prediction = None
        if self.future_weather_mode in {
            "persistence", "learned", "seasonal", "seasonal_weighted"
        }:
            future_weather = self._causal_future_weather(weather)
        if self.future_weather_mode == "learned":
            causal_weather_prediction = future_weather
            if "diagnostic_future_weather_override" in batch:
                if self.training:
                    raise RuntimeError("Oracle weather override is evaluation-only")
                future_weather = batch["diagnostic_future_weather_override"]
        if self.latent_forcing:
            global_forcing, local_forcing = self._initial_latent_forcing(weather)
        if self.factorized_forcing:
            transport_forcing, source_forcing = self._initial_factorized_forcing(weather)
            previous_weather = weather[:, -1]
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
        factorized_weather_predictions = []
        for step in range(self.horizon):
            state = state_history[-1]
            if self.latent_forcing:
                state_mean = state.mean(-1, keepdim=True)
                global_forcing = self.global_forcing_cell(state_mean, global_forcing)
                global_nodes = global_forcing[:, None].expand(-1, stations, -1)
                residual_forcing_input = torch.cat((
                    (state - state_mean)[..., None], global_nodes,
                ), -1)
                local_forcing = self.local_forcing_cell(
                    residual_forcing_input.reshape(batch_size * stations, -1),
                    local_forcing.reshape(batch_size * stations, -1),
                ).reshape(batch_size, stations, -1)
                common_forcing_step = global_forcing
                local_forcing_step = local_forcing
            elif self.factorized_forcing:
                horizon_step = self.horizon_embedding.weight[step]
                horizon_nodes = horizon_step[None, None].expand(
                    batch_size, stations, -1
                )
                transport_forcing = self.transport_forcing_cell(
                    horizon_nodes.reshape(batch_size * stations, -1),
                    transport_forcing.reshape(batch_size * stations, -1),
                ).reshape(batch_size, stations, -1)
                source_forcing = self.source_forcing_cell(
                    horizon_nodes.reshape(batch_size * stations, -1),
                    source_forcing.reshape(batch_size * stations, -1),
                ).reshape(batch_size, stations, -1)
                previous_weather = self._factorized_weather_step(
                    previous_weather, transport_forcing, source_forcing
                )
                factorized_weather_predictions.append(previous_weather)
                weather_step = previous_weather
                common_forcing_step = source_forcing.mean(1)
                local_forcing_step = source_forcing
            else:
                weather_step = future_weather[:, step]
                common_forcing_step = weather_step.mean(1)
                local_forcing_step = weather_step
            auxiliary_step = auxiliary[:, step]
            if self.use_month:
                month = self.month_embedding(batch["future_month"][:, step])
            else:
                month = x.new_zeros(batch_size, self.month_embedding.embedding_dim)
            month_nodes = month[:, None].expand(-1, stations, -1)
            common_features = torch.cat((
                state.mean(-1, keepdim=True), common_forcing_step,
                auxiliary_step.mean(1), month,
            ), -1)
            common_hidden = self.common_cell(common_features, common_hidden)

            if self.latent_forcing:
                innovation_now = self._spatial_innovation(state)
                innovation_lag = torch.zeros_like(innovation_now)
                transport_context = self.latent_transport_signal(local_forcing_step)
            elif self.factorized_forcing:
                innovation_now = self._wind_innovation(
                    state, weather_step[..., 4], weather_step[..., 5]
                )
                innovation_lag = torch.zeros_like(innovation_now)
                transport_context = weather_step[..., 3:4]
            else:
                innovation_now = self._wind_innovation(
                    state, weather_step[..., 4], weather_step[..., 5]
                )
                innovation_lag = self._wind_innovation(
                    state_history[-2], weather_step[..., 4], weather_step[..., 5]
                )
                transport_context = weather_step[..., 3:4]
            if not self.use_lagged_transport:
                innovation_lag = torch.zeros_like(innovation_lag)
            if not self.use_transport:
                innovation_now = torch.zeros_like(innovation_now)
                innovation_lag = torch.zeros_like(innovation_lag)
            residual_state = state - state.mean(-1, keepdim=True)
            local_features = torch.cat((
                residual_state[..., None], local_forcing_step, auxiliary_step,
                month_nodes, station,
            ), -1)
            if self.latent_forcing or self.factorized_forcing:
                local_recurrent = torch.cat((
                    local_features, innovation_now[..., None],
                ), -1)
            else:
                local_recurrent = torch.cat((
                    local_features, innovation_now[..., None], innovation_lag[..., None],
                ), -1)
            local_hidden = self.local_cell(
                local_recurrent.reshape(batch_size * stations, -1),
                local_hidden.reshape(batch_size * stations, -1),
            ).reshape(batch_size, stations, -1)

            if self.latent_forcing or self.factorized_forcing:
                transport_input = torch.cat((
                    local_hidden, innovation_now[..., None], transport_context,
                ), -1)
            else:
                transport_input = torch.cat((
                    local_hidden, innovation_now[..., None], innovation_lag[..., None],
                    transport_context,
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
        if self.future_weather_mode == "learned":
            result["weather_prediction"] = causal_weather_prediction
        if self.factorized_forcing:
            result["weather_prediction"] = torch.stack(
                factorized_weather_predictions, 1
            )
        if gates:
            result["event_gate"] = torch.stack(gates, 1)
        return result
