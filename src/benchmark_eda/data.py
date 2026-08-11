"""Small data helpers used only by the benchmark EDA notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
]
POLLUTANT_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_FEATURES = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "WD_SIN", "WD_COS"]
WIND_ANGLE = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def load_raw_frames(raw_dir: str | Path) -> pd.DataFrame:
    """Load the 12 raw UCI station CSVs without model preprocessing."""
    files = sorted(Path(raw_dir).glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No station CSVs in {raw_dir}. Run: python scripts/download_data.py"
        )
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
        if "station" not in frame:
            frame["station"] = path.stem.split("_")[2]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "station"])


def _haversine_graph(coordinates: np.ndarray, k: int):
    """Return nearest neighbours, distance, and source-to-target directions."""
    if not 1 <= k < len(coordinates):
        raise ValueError(f"neighbors must be in [1, {len(coordinates) - 1}]")
    lon = np.radians(coordinates[:, 0]); lat = np.radians(coordinates[:, 1])
    dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    neighbours = np.argsort(distance, axis=1)[:, 1:k + 1]
    neighbour_distance = np.take_along_axis(distance, neighbours, axis=1)
    target_lon, target_lat = lon[:, None], lat[:, None]
    source_lon, source_lat = lon[neighbours], lat[neighbours]
    east = (target_lon - source_lon) * np.cos((target_lat + source_lat) / 2)
    north = target_lat - source_lat
    norm = np.sqrt(east ** 2 + north ** 2).clip(1e-8)
    direction = np.stack([east / norm, north / norm], axis=-1)
    return (neighbours.astype(np.int64), neighbour_distance.astype(np.float32),
            direction.astype(np.float32))
