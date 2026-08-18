"""Contract and leakage checks for forecast-origin meteorological archives."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .data import KnowAirDataModule


REQUIRED_ARRAYS = {
    "forecast_start",
    "issue_time_ns",
    "valid_time_ns",
    "weather",
    "weather_feature_names",
    "metadata_json",
}


@dataclass(frozen=True)
class ForecastArchive:
    """An immutable meteorological forecast archive aligned to KnowAir.

    The NPZ is intentionally simple and safe to load with ``allow_pickle=False``.
    Required arrays are:

    - ``forecast_start``: KnowAir integer origin index, shape ``[S]``;
    - ``issue_time_ns``: forecast publication time in UTC nanoseconds, ``[S]``;
    - ``valid_time_ns``: valid times in UTC nanoseconds, ``[S, H]``;
    - ``weather``: forecast values, ``[S, H, N, W]``;
    - ``weather_feature_names``: names for the final axis;
    - ``metadata_json``: scalar JSON with non-empty ``source`` and
      ``model_version`` and ``time_basis: \"UTC\"`` fields.

    A forecast is accepted only when it was issued no later than its origin and
    every valid time exactly matches the corresponding KnowAir future lead.
    """

    path: Path
    forecast_start: np.ndarray
    issue_time_ns: np.ndarray
    valid_time_ns: np.ndarray
    weather: np.ndarray
    weather_feature_names: tuple[str, ...]
    metadata: dict[str, object]

    @classmethod
    def load(cls, path: str | Path) -> "ForecastArchive":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            missing = REQUIRED_ARRAYS - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            try:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError("metadata_json must be a scalar JSON object") from error
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must decode to an object")
            required_metadata = ("source", "model_version", "time_basis")
            absent = [key for key in required_metadata if not metadata.get(key)]
            if absent:
                raise ValueError(f"metadata_json is missing non-empty fields: {absent}")
            if metadata["time_basis"] != "UTC":
                raise ValueError("metadata_json.time_basis must be UTC")
            result = cls(
                path=path,
                forecast_start=np.asarray(archive["forecast_start"], dtype=np.int64),
                issue_time_ns=np.asarray(archive["issue_time_ns"], dtype=np.int64),
                valid_time_ns=np.asarray(archive["valid_time_ns"], dtype=np.int64),
                weather=np.asarray(archive["weather"], dtype=np.float32),
                weather_feature_names=tuple(np.asarray(archive["weather_feature_names"]).astype(str)),
                metadata=metadata,
            )
        result._validate_shape()
        return result

    def _validate_shape(self) -> None:
        samples = len(self.forecast_start)
        if self.forecast_start.ndim != 1 or self.issue_time_ns.shape != (samples,):
            raise ValueError("forecast_start and issue_time_ns must have shape [S]")
        if self.valid_time_ns.ndim != 2 or self.valid_time_ns.shape[0] != samples:
            raise ValueError("valid_time_ns must have shape [S, H]")
        if self.weather.ndim != 4 or self.weather.shape[:2] != self.valid_time_ns.shape:
            raise ValueError("weather must have shape [S, H, N, W] matching valid_time_ns")
        if self.weather.shape[-1] != len(self.weather_feature_names):
            raise ValueError("weather_feature_names does not match weather's final axis")
        if not self.weather_feature_names or len(set(self.weather_feature_names)) != len(self.weather_feature_names):
            raise ValueError("weather_feature_names must be non-empty and unique")
        if not np.isfinite(self.weather).all():
            raise ValueError("weather forecast contains non-finite values")
        if len(np.unique(self.forecast_start)) != samples:
            raise ValueError("forecast_start contains duplicate origins")

    def validate_for_validation(
        self,
        data: KnowAirDataModule,
        origins: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Return archive row indices after strict validation-only checks."""
        origins = np.asarray(origins, dtype=np.int64)
        if self.weather.shape[1] != horizon:
            raise ValueError(
                f"archive horizon {self.weather.shape[1]} does not equal requested {horizon}"
            )
        if self.weather.shape[2] != data.stations:
            raise ValueError("archive station axis does not match KnowAir")
        unknown = set(self.weather_feature_names) - set(data.weather_feature_names)
        if unknown:
            raise ValueError(f"archive has unknown weather features: {sorted(unknown)}")
        if origins.min() < data.boundaries["val"][0] or origins.max() + horizon > data.boundaries["val"][1]:
            raise ValueError("requested origins are not wholly inside validation")
        lookup = {int(origin): index for index, origin in enumerate(self.forecast_start)}
        missing = [int(origin) for origin in origins if int(origin) not in lookup]
        if missing:
            raise ValueError(f"archive does not cover {len(missing)} required validation origins")
        rows = np.asarray([lookup[int(origin)] for origin in origins], dtype=np.int64)
        origin_time_ns = data.timestamps[origins].asi8
        if np.any(self.issue_time_ns[rows] > origin_time_ns):
            raise ValueError("future-information leakage: issue_time_ns is after forecast origin")
        expected_valid = data.timestamps.asi8[
            origins[:, None] + np.arange(horizon, dtype=np.int64)[None]
        ]
        if not np.array_equal(self.valid_time_ns[rows], expected_valid):
            raise ValueError("valid_time_ns does not exactly match forecast-origin leads")
        return rows
