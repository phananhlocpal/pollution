"""Minimal leakage-safe KnowAir loader used by the Quantile Router."""

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
    """Expose only the causal PM/history metadata required by the router."""

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
