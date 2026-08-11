"""KnowAir adapter for the canonical common/local experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


RAW_FEATURES = (
    "100m_u_wind", "100m_v_wind", "2m_dewpoint", "2m_temperature",
    "boundary_layer_height", "k_index", "relative_humidity_950",
    "relative_humidity_975", "specific_humidity_950", "surface_pressure",
    "temperature_925", "temperature_950", "total_precipitation",
    "u_wind_950", "v_wind_950", "vertical_velocity_950",
    "vorticity_950", "PM2.5",
)
FEATURES = (
    "PM2.5", "2m_temperature", "surface_pressure", "relative_humidity_950",
    "100m_wind_speed_kmh", "100m_wind_direction_sin",
    "100m_wind_direction_cos",
)
AUXILIARY_FEATURES = (
    "2m_dewpoint", "total_precipitation", "boundary_layer_height",
    "ventilation", "dewpoint_deficit",
)


@dataclass
class Panel:
    values: np.ndarray
    physical: np.ndarray
    split_points: tuple[int, int]
    mean: np.ndarray
    std: np.ndarray
    stations: list[str]
    feature_names: tuple[str, ...] = FEATURES
    cadence_hours: int = 3
    auxiliary: np.ndarray | None = None
    auxiliary_mean: np.ndarray | None = None
    auxiliary_std: np.ndarray | None = None
    auxiliary_feature_names: tuple[str, ...] = AUXILIARY_FEATURES


def load_panel(root: str | Path = ".") -> Panel:
    """Load KnowAir and fit scaling on the chronological training half only."""
    root = Path(root)
    raw = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    if raw.ndim != 3 or raw.shape[1:] != (184, 18):
        raise ValueError(f"Expected KnowAir [T,184,18], got {raw.shape}")
    u = np.asarray(raw[..., RAW_FEATURES.index("100m_u_wind")], dtype=np.float64)
    v = np.asarray(raw[..., RAW_FEATURES.index("100m_v_wind")], dtype=np.float64)
    direction = np.degrees(np.arctan2(-u, -v)) % 360.0
    radians = np.deg2rad(direction)
    physical = np.stack((
        np.asarray(raw[..., RAW_FEATURES.index("PM2.5")]),
        np.asarray(raw[..., RAW_FEATURES.index("2m_temperature")]),
        np.asarray(raw[..., RAW_FEATURES.index("surface_pressure")]),
        np.asarray(raw[..., RAW_FEATURES.index("relative_humidity_950")]),
        3.6 * np.hypot(u, v), np.sin(radians), np.cos(radians),
    ), axis=-1).astype(np.float64)
    if not np.isfinite(physical).all():
        raise ValueError("KnowAir tensor contains non-finite values")
    train_end, val_end = int(len(physical) * .50), int(len(physical) * .75)
    train = physical[:train_end].reshape(-1, len(FEATURES))
    mean = train.mean(0, dtype=np.float64)
    std = train.std(0, dtype=np.float64)
    if np.any(std < 1e-8):
        raise ValueError("Degenerate train-fitted feature scale")
    values = ((physical - mean) / std).astype(np.float32)
    dewpoint = np.asarray(raw[..., RAW_FEATURES.index("2m_dewpoint")], dtype=np.float64)
    temperature = np.asarray(raw[..., RAW_FEATURES.index("2m_temperature")], dtype=np.float64)
    pbl = np.asarray(raw[..., RAW_FEATURES.index("boundary_layer_height")], dtype=np.float64)
    precipitation = np.asarray(raw[..., RAW_FEATURES.index("total_precipitation")], dtype=np.float64)
    auxiliary_physical = np.stack((
        dewpoint, precipitation, pbl, np.hypot(u, v) * pbl, temperature - dewpoint,
    ), axis=-1)
    auxiliary_train = auxiliary_physical[:train_end].reshape(-1, len(AUXILIARY_FEATURES))
    auxiliary_mean = auxiliary_train.mean(0, dtype=np.float64)
    auxiliary_std = auxiliary_train.std(0, dtype=np.float64)
    if np.any(auxiliary_std < 1e-8):
        raise ValueError("Degenerate train-fitted auxiliary feature scale")
    auxiliary = ((auxiliary_physical - auxiliary_mean) / auxiliary_std).astype(np.float32)
    stations = np.loadtxt(
        root / "data/benchmarks/knowair/city.txt", usecols=(1,), dtype=str
    ).tolist()
    return Panel(
        values, physical, (train_end, val_end), mean, std, stations,
        auxiliary=auxiliary, auxiliary_mean=auxiliary_mean, auxiliary_std=auxiliary_std,
    )


class CommonLocalWindowDataset(Dataset):
    """72-hour history and future weather mapped to a 72-hour PM forecast."""

    def __init__(self, panel: Panel, split: str, max_samples: int | None = None):
        train_end, val_end = panel.split_points
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        left, right = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, len(panel.values)),
        }[split]
        history = horizon = 24
        count = right - left - history - horizon + 1
        starts = np.arange(left, left + count, dtype=np.int64)
        if max_samples is not None and len(starts) > max_samples:
            starts = starts[np.linspace(0, len(starts) - 1, max_samples, dtype=int)]
        self.panel, self.split, self.starts = panel, split, starts
        self.history, self.horizon = history, horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index]); future_start = start + self.history
        history = self.panel.values[start:future_start]
        future = self.panel.values[future_start:future_start + self.horizon]
        sample = {
            "x": torch.from_numpy(history),
            "future_weather": torch.from_numpy(future[..., 1:]),
            "y": torch.from_numpy(future[..., 0]),
            "forecast_start": torch.tensor(future_start, dtype=torch.long),
        }
        if self.panel.auxiliary is not None:
            sample["future_auxiliary"] = torch.from_numpy(
                self.panel.auxiliary[future_start:future_start + self.horizon]
            )
        return sample

