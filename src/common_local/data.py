"""KnowAir adapter for the canonical common/local experiment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

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
    coordinates: np.ndarray | None = None


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
    coordinates = np.loadtxt(
        root / "data/benchmarks/knowair/city.txt", usecols=(2, 3), dtype=float
    )
    return Panel(
        values, physical, (train_end, val_end), mean, std, stations,
        auxiliary=auxiliary, auxiliary_mean=auxiliary_mean, auxiliary_std=auxiliary_std,
        coordinates=coordinates,
    )


def load_standard_panel(path: str | Path, expected_stations: int | None = None) -> Panel:
    """Load an external lockbox from the documented, dataset-neutral NPZ contract.

    Required arrays are ``target[T,N]``, ``weather[T,N,W]`` and
    ``coordinates[N,2]`` (longitude, latitude). Scaling is fitted on the first
    70% only; the chronological split is 7:1:2 as reported for AirDDE's external
    benchmarks. The archive must already encode circular wind direction as
    separate sine/cosine fields and put wind speed/sine/cosine at weather
    columns 3/4/5, respectively.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = {"target", "weather", "coordinates"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        target = np.asarray(archive["target"], dtype=np.float64)
        weather = np.asarray(archive["weather"], dtype=np.float64)
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64)
        station_ids = (
            np.asarray(archive["station_ids"]).astype(str).tolist()
            if "station_ids" in archive.files else [str(i) for i in range(target.shape[1])]
        )
        cadence_hours = int(np.asarray(archive["cadence_hours"]).item()) if "cadence_hours" in archive.files else 1
    if target.ndim != 2 or weather.ndim != 3 or weather.shape[:2] != target.shape:
        raise ValueError(f"Expected target [T,N] and weather [T,N,W], got {target.shape}, {weather.shape}")
    if expected_stations is not None and target.shape[1] != expected_stations:
        raise ValueError(f"Expected {expected_stations} stations, got {target.shape[1]}")
    if weather.shape[-1] < 6:
        raise ValueError("The frozen operator requires weather speed/sin/cos columns 3/4/5")
    if coordinates.shape != (target.shape[1], 2):
        raise ValueError(f"Expected coordinates {(target.shape[1], 2)}, got {coordinates.shape}")
    physical = np.concatenate((target[..., None], weather), axis=-1)
    if not np.isfinite(physical).all() or not np.isfinite(coordinates).all():
        raise ValueError("External panel contains non-finite values")
    train_end, val_end = int(len(physical) * .70), int(len(physical) * .80)
    train = physical[:train_end].reshape(-1, physical.shape[-1])
    mean, std = train.mean(0), train.std(0)
    if np.any(std < 1e-8):
        raise ValueError("Degenerate train-fitted feature scale")
    values = ((physical - mean) / std).astype(np.float32)
    feature_names = ("target",) + tuple(f"weather_{i}" for i in range(weather.shape[-1]))
    return Panel(
        values=values, physical=physical, split_points=(train_end, val_end),
        mean=mean, std=std, stations=station_ids, feature_names=feature_names,
        auxiliary=None, auxiliary_mean=None, auxiliary_std=None,
        auxiliary_feature_names=(), coordinates=coordinates, cadence_hours=cadence_hours,
    )


GAGNN_WEATHER_COLUMNS = (3, 1, 2, 4, 5, 6, 0)


@lru_cache(maxsize=2)
def load_gagnn_metadata(directory: str | Path):
    """Fit train-only scaling for the official pre-windowed GAGNN release."""
    directory = Path(directory).resolve()
    train_x = np.load(directory / "train_x.npy", mmap_mode="r")
    train_y = np.load(directory / "train_y.npy", mmap_mode="r")
    coordinates = np.load(directory / "loc_filled.npy", allow_pickle=True).astype(float)
    if train_x.ndim != 4 or train_x.shape[1:] != (24, 209, 8):
        raise ValueError(f"Expected official GAGNN train_x [S,24,209,8], got {train_x.shape}")
    if train_y.shape[1:] != (6, 209) or coordinates.shape != (209, 2):
        raise ValueError("Official GAGNN target/coordinate shapes do not match 209-city protocol")
    order = (7,) + GAGNN_WEATHER_COLUMNS
    total = np.zeros(8, dtype=np.float64)
    square = np.zeros(8, dtype=np.float64)
    count = 0
    for left in range(0, len(train_x), 128):
        chunk = np.asarray(train_x[left:left + 128, ..., order], dtype=np.float64)
        total += chunk.sum(axis=(0, 1, 2))
        square += np.square(chunk).sum(axis=(0, 1, 2))
        count += np.prod(chunk.shape[:-1])
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 0))
    if np.any(std < 1e-8):
        raise ValueError("Degenerate GAGNN train-fitted feature scale")
    return SimpleNamespace(
        mean=mean, std=std, stations=[str(i) for i in range(209)],
        coordinates=coordinates, weather_dim=7, target_threshold=np.zeros(209),
        source="official_gagnn_pre_windowed", history=24, horizon=6, cadence_hours=1,
    )


class GAGNNWindowDataset(Dataset):
    """Official 209-city China-AQI windows, without reconstructing/leaking splits."""

    def __init__(self, directory, split, metadata, max_samples=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        directory = Path(directory)
        self.x = np.load(directory / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(directory / f"{split}_y.npy", mmap_mode="r")
        self.indices = np.arange(len(self.x), dtype=np.int64)
        if max_samples is not None and len(self.indices) > max_samples:
            self.indices = self.indices[np.linspace(0, len(self.indices) - 1, max_samples, dtype=int)]
        self.metadata = metadata

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        raw = np.asarray(self.x[source], dtype=np.float64)
        physical = np.concatenate((raw[..., 7:8], raw[..., GAGNN_WEATHER_COLUMNS]), axis=-1)
        history = ((physical - self.metadata.mean) / self.metadata.std).astype(np.float32)
        target = ((np.asarray(self.y[source]) - self.metadata.mean[0]) / self.metadata.std[0]).astype(np.float32)
        # The official release exposes historical covariates only. This placeholder
        # is deliberately ignored by persistence/learned causal weather modes.
        future_weather = np.zeros((6, 209, 7), dtype=np.float32)
        return {
            "x": torch.from_numpy(history),
            "future_weather": torch.from_numpy(future_weather),
            "y": torch.from_numpy(target),
            "forecast_start": torch.tensor(source + 24, dtype=torch.long),
        }


class CommonLocalWindowDataset(Dataset):
    """72-hour history and future weather mapped to a 72-hour PM forecast."""

    def __init__(
        self, panel: Panel, split: str, max_samples: int | None = None,
        history: int = 24, horizon: int = 24,
    ):
        train_end, val_end = panel.split_points
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        left, right = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, len(panel.values)),
        }[split]
        count = right - left - history - horizon + 1
        if count <= 0:
            raise ValueError(
                f"Split {split} has {right-left} steps, fewer than history+horizon={history+horizon}"
            )
        starts = np.arange(left, left + count, dtype=np.int64)
        if max_samples is not None and len(starts) > max_samples:
            starts = starts[np.linspace(0, len(starts) - 1, max_samples, dtype=int)]
        self.panel, self.split, self.starts = panel, split, starts
        self.history, self.horizon = history, horizon
        timestamps = np.datetime64("2015-01-01T00") + (
            np.arange(len(panel.values)) * panel.cadence_hours
        ).astype("timedelta64[h]")
        self.month_by_time = timestamps.astype("datetime64[M]").astype(np.int64) % 12

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index]); future_start = start + self.history
        history = self.panel.values[start:future_start]
        future = self.panel.values[future_start:future_start + self.horizon]
        sample = {
            "x": torch.from_numpy(history),
            "future_weather": torch.from_numpy(future[..., 1:]),
            "future_weather_target": torch.from_numpy(future[..., 1:]),
            "y": torch.from_numpy(future[..., 0]),
            "forecast_start": torch.tensor(future_start, dtype=torch.long),
        }
        if self.panel.auxiliary is not None:
            sample["future_auxiliary"] = torch.from_numpy(
                self.panel.auxiliary[future_start:future_start + self.horizon]
            )
            sample["future_month"] = torch.from_numpy(
                self.month_by_time[future_start:future_start + self.horizon].copy()
            ).long()
        return sample

