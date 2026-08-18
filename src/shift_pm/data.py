"""Leakage-safe KnowAir loader shared by the retained diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FEATURES = (
    "u100", "v100", "dewpoint", "temperature", "pbl", "k_index", "rh950",
    "rh975", "specific_humidity950", "pressure", "temperature925",
    "temperature950", "precipitation", "u950", "v950",
    "vertical_velocity950", "vorticity950", "pm25",
)


@dataclass(frozen=True)
class QuantileRouterProtocol:
    history_steps: int = 24
    horizon: int = 24
    train_fraction: float = 0.50
    validation_fraction: float = 0.25
    cadence_hours: int = 3

    def boundaries(self, length: int) -> dict[str, tuple[int, int]]:
        train_end = int(length * self.train_fraction)
        validation_end = int(
            length * (self.train_fraction + self.validation_fraction)
        )
        return {
            "train": (0, train_end),
            "val": (train_end, validation_end),
            "test": (validation_end, length),
        }


@dataclass(frozen=True)
class PMTransform:
    log_climatology: np.ndarray  # [12, 8, station]
    anomaly_scale: np.ndarray  # [station]

    @classmethod
    def fit(
        cls,
        pm: np.ndarray,
        timestamps: pd.DatetimeIndex,
        train_end: int,
    ) -> "PMTransform":
        train_pm = pm[:train_end].astype(np.float64, copy=True)
        train_pm[train_pm <= 0] = np.nan
        train_log = np.log1p(train_pm)
        station_fallback = np.nanmean(train_log, axis=0)
        global_fallback = float(np.nanmean(train_log))
        station_fallback = np.where(
            np.isfinite(station_fallback), station_fallback, global_fallback
        )
        train_time = timestamps[:train_end]
        climatology = np.empty((12, 8, pm.shape[1]), dtype=np.float64)
        for month in range(1, 13):
            for slot in range(8):
                selected = (
                    (train_time.month.to_numpy() == month)
                    & (train_time.hour.to_numpy() // 3 == slot)
                )
                if selected.any():
                    with np.errstate(invalid="ignore"):
                        value = np.nanmean(train_log[selected], axis=0)
                else:
                    value = station_fallback
                climatology[month - 1, slot] = np.where(
                    np.isfinite(value), value, station_fallback
                )
        baseline = climatology[
            train_time.month.to_numpy() - 1,
            train_time.hour.to_numpy() // 3,
        ]
        with np.errstate(invalid="ignore"):
            scale = np.nanstd(train_log - baseline, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        return cls(climatology.astype(np.float32), scale.astype(np.float32))

    def baseline(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        return self.log_climatology[
            timestamps.month.to_numpy() - 1,
            timestamps.hour.to_numpy() // 3,
        ]


class KnowAirDataModule:
    """Expose causal history and observed data for validation-only diagnostics.

    ``observed_weather`` is deliberately explicit about forecast origins and
    horizon.  It is useful for perfect-prognosis diagnostics, but callers must
    not pass its output off as forecast-origin information.
    """

    def __init__(
        self,
        root: str | Path = "data/benchmarks/knowair",
        protocol: QuantileRouterProtocol | None = None,
    ) -> None:
        self.root = Path(root)
        self.protocol = protocol or QuantileRouterProtocol()
        raw_path = self.root / "KnowAir.npy"
        city_path = self.root / "city.txt"
        if not raw_path.is_file() or not city_path.is_file():
            raise FileNotFoundError(
                f"KnowAir.npy and city.txt are required under {self.root}"
            )
        raw = np.load(raw_path, mmap_mode="r")
        if raw.ndim != 3 or raw.shape[-1] != len(RAW_FEATURES):
            raise ValueError(f"unexpected KnowAir shape {raw.shape}")
        self._raw = raw
        self.length, self.stations = raw.shape[:2]
        self.timestamps = pd.date_range(
            "2015-01-01",
            periods=self.length,
            freq=f"{self.protocol.cadence_hours}h",
        )
        self.boundaries = self.protocol.boundaries(self.length)
        self.coordinates = np.loadtxt(city_path, usecols=(2, 3)).astype(np.float32)
        if self.coordinates.shape != (self.stations, 2):
            raise ValueError("city.txt station count does not match KnowAir.npy")

        pm = np.asarray(raw[..., RAW_FEATURES.index("pm25")], dtype=np.float32)
        pm[pm <= 0] = np.nan
        self.transform = PMTransform.fit(
            pm, self.timestamps, self.boundaries["train"][1]
        )
        self.log_climatology = self.transform.baseline(self.timestamps).astype(
            np.float32
        )
        self.pm_anomaly = (
            (np.log1p(pm) - self.log_climatology)
            / self.transform.anomaly_scale[None]
        ).astype(np.float32)
        self.pm_anomaly = np.nan_to_num(
            self.pm_anomaly, nan=0.0, posinf=0.0, neginf=0.0
        )

    @property
    def weather_feature_names(self) -> tuple[str, ...]:
        """Raw meteorological channel names, in their archive order."""
        return tuple(name for name in RAW_FEATURES if name != "pm25")

    @property
    def timestamp_basis(self) -> str:
        """Interpretation required by the forecast-archive contract."""
        return "UTC"

    def observed_weather(
        self,
        origins: np.ndarray,
        horizon: int,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        """Return realized future weather as ``[origin, lead, station, feature]``.

        This accessor exists so an audit can calculate forecast-weather error.
        It is *not* a source of deployable covariates.  Bounds are checked here
        so callers cannot silently cross a chronological split.
        """
        origins = np.asarray(origins, dtype=np.int64)
        if origins.ndim != 1 or not len(origins):
            raise ValueError("origins must be a non-empty one-dimensional array")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if origins.min() < 0 or origins.max() + horizon > self.length:
            raise ValueError("requested future weather is outside KnowAir")
        unknown = set(feature_names) - set(self.weather_feature_names)
        if unknown:
            raise ValueError(f"unknown meteorological features: {sorted(unknown)}")
        indices = [RAW_FEATURES.index(name) for name in feature_names]
        steps = origins[:, None] + np.arange(horizon, dtype=np.int64)[None]
        return np.asarray(self._raw[steps, :, :][..., indices], dtype=np.float32)
