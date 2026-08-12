#!/usr/bin/env python3
"""Create a leakage-audited 3-hour Beijing Multi-Site external panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WIND_DEGREES = {
    name: index * 22.5
    for index, name in enumerate(
        ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    )
}
COORDINATE_ALIASES = {"changping": "pingchang"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_fill(values: pd.Series, training_end: int) -> np.ndarray:
    """Forward-fill a channel; use a train-only median only for a leading gap."""
    numeric = pd.to_numeric(values, errors="coerce")
    fallback = float(numeric.iloc[:training_end].median())
    if not np.isfinite(fallback):
        raise ValueError(f"No finite training value for {values.name}")
    return numeric.ffill().fillna(fallback).to_numpy(dtype=np.float64)


def relative_humidity(temperature: np.ndarray, dewpoint: np.ndarray) -> np.ndarray:
    """Magnus approximation for relative humidity in percent."""
    vapor = np.exp((17.625 * dewpoint) / (243.04 + dewpoint))
    saturation = np.exp((17.625 * temperature) / (243.04 + temperature))
    return np.clip(100.0 * vapor / saturation, 0.0, 100.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default="data/raw/PRSA_Data_20130301-20170228"
    )
    parser.add_argument(
        "--coordinates", default="data/benchmarks/beijing_kdd/beijing_station_coords.csv"
    )
    parser.add_argument(
        "--output", default="data/processed/beijing_multisite_3h.npz"
    )
    parser.add_argument(
        "--manifest", default="paper/artifacts/beijing_multisite_protocol.json"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("PRSA_Data_*_20130301-20170228.csv"))
    if len(files) != 12:
        raise ValueError(f"Expected 12 UCI station files, found {len(files)}")
    raw_frames = [pd.read_csv(path) for path in files]
    timestamps = pd.to_datetime(raw_frames[0][["year", "month", "day", "hour"]])
    if not all(
        pd.to_datetime(frame[["year", "month", "day", "hour"]]).equals(timestamps)
        for frame in raw_frames[1:]
    ):
        raise ValueError("Station timestamps are not exactly aligned")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must be unique and chronological")

    hourly_train_end = int(len(timestamps) * 0.70)
    targets, masks, weather, station_ids = [], [], [], []
    missing_by_station = {}
    for path, frame in zip(files, raw_frames):
        station = str(frame["station"].iloc[0])
        station_ids.append(station)
        observed = frame["PM2.5"].notna().to_numpy() & (frame["PM2.5"].fillna(0).to_numpy() >= 1e-4)
        target = causal_fill(frame["PM2.5"], hourly_train_end)
        temperature = causal_fill(frame["TEMP"], hourly_train_end)
        pressure = causal_fill(frame["PRES"], hourly_train_end)
        dewpoint = causal_fill(frame["DEWP"], hourly_train_end)
        speed = causal_fill(frame["WSPM"], hourly_train_end) * 3.6
        direction = frame["wd"].map(WIND_DEGREES)
        direction = causal_fill(direction, hourly_train_end)
        radians = np.deg2rad(direction)
        station_weather = np.stack((
            temperature,
            pressure,
            relative_humidity(temperature, dewpoint),
            speed,
            np.sin(radians),
            np.cos(radians),
        ), axis=-1)
        targets.append(target)
        masks.append(observed)
        weather.append(station_weather)
        missing_by_station[station] = {
            "pm25_missing_fraction_hourly": float(1.0 - observed.mean()),
            "weather_rows_with_missing_fraction_hourly": float(
                frame[["TEMP", "PRES", "DEWP", "wd", "WSPM"]].isna().any(axis=1).mean()
            ),
        }

    target = np.stack(targets, axis=1)[::3]
    target_valid = np.stack(masks, axis=1)[::3]
    weather_values = np.stack(weather, axis=1)[::3]
    timestamp_values = timestamps.to_numpy(dtype="datetime64[h]")[::3]
    if not np.isfinite(target).all() or not np.isfinite(weather_values).all():
        raise ValueError("Causal filling left non-finite values")

    coordinates_frame = pd.read_csv(args.coordinates).set_index("station")
    coordinates = []
    coordinate_rows = []
    for station in station_ids:
        key = COORDINATE_ALIASES.get(station.lower(), station.lower()) + "_aq"
        if key not in coordinates_frame.index:
            raise KeyError(f"No coordinate for {station} ({key})")
        row = coordinates_frame.loc[key]
        coordinates.append((float(row["longitude"]), float(row["latitude"])))
        coordinate_rows.append(key)
    coordinates = np.asarray(coordinates, dtype=np.float64)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target=target.astype(np.float32),
        target_valid=target_valid,
        weather=weather_values.astype(np.float32),
        coordinates=coordinates,
        station_ids=np.asarray(station_ids),
        timestamps=timestamp_values,
        cadence_hours=np.asarray(3, dtype=np.int64),
    )
    train_end, val_end = int(len(target) * .70), int(len(target) * .80)
    manifest = {
        "dataset": "UCI Beijing Multi-Site Air Quality",
        "doi": "10.24432/C5RK5G",
        "source_url": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "license": "CC BY 4.0",
        "raw_period": [str(timestamps.iloc[0]), str(timestamps.iloc[-1])],
        "cadence_hours": 3,
        "shape": {"time": len(target), "stations": len(station_ids), "weather": 6},
        "weather_order": [
            "temperature_celsius", "surface_pressure_hpa", "relative_humidity_percent",
            "wind_speed_kmh", "wind_direction_sin", "wind_direction_cos",
        ],
        "relative_humidity": "Magnus approximation from observed temperature and dew point",
        "unused_available_variable": "rainfall",
        "missing_data_policy": (
            "Target validity is retained as a mask for loss and metrics. Input gaps are "
            "causally forward-filled; a train-only median is used only for leading gaps."
        ),
        "split": {
            "train": [0, train_end],
            "validation": [train_end, val_end],
            "test": [val_end, len(target)],
            "ratio": "7:1:2 chronological",
            "test_accessed": False,
        },
        "observed_target_fraction": {
            "train": float(target_valid[:train_end].mean()),
            "validation": float(target_valid[train_end:val_end].mean()),
            "test": float(target_valid[val_end:].mean()),
        },
        "missing_by_station": missing_by_station,
        "station_ids": station_ids,
        "coordinate_rows": coordinate_rows,
        "input_sha256": {path.name: sha256(path) for path in files},
        "coordinate_file_sha256": sha256(Path(args.coordinates)),
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
