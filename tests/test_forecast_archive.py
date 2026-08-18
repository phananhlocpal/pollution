from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shift_pm import ForecastArchive, KnowAirDataModule
from shift_pm.data import RAW_FEATURES


def make_data(root: Path) -> KnowAirDataModule:
    root.mkdir(parents=True)
    raw = np.ones((240, 2, len(RAW_FEATURES)), dtype=np.float64)
    raw[..., RAW_FEATURES.index("pm25")] = 25.0
    np.save(root / "KnowAir.npy", raw)
    (root / "city.txt").write_text("0 A 100 30\n1 B 101 31\n", encoding="utf-8")
    return KnowAirDataModule(root)


def write_archive(path: Path, data: KnowAirDataModule, issue_after_origin: bool = False) -> None:
    origins = np.array([120, 121], dtype=np.int64)
    horizon = 24
    names = np.array(["temperature", "u100"])
    issue = data.timestamps.asi8[origins].copy()
    if issue_after_origin:
        issue[0] += 1
    valid = data.timestamps.asi8[origins[:, None] + np.arange(horizon)[None]]
    weather = data.observed_weather(origins, horizon, tuple(names))
    np.savez(
        path,
        forecast_start=origins,
        issue_time_ns=issue,
        valid_time_ns=valid,
        weather=weather,
        weather_feature_names=names,
        metadata_json=json.dumps({
            "source": "unit-test", "model_version": "1", "time_basis": "UTC"
        }),
    )


def test_archive_accepts_origin_safe_validation_weather(tmp_path: Path) -> None:
    data = make_data(tmp_path / "knowair")
    archive_path = tmp_path / "forecast.npz"
    write_archive(archive_path, data)
    archive = ForecastArchive.load(archive_path)
    rows = archive.validate_for_validation(data, np.array([120, 121]), horizon=24)
    np.testing.assert_array_equal(rows, [0, 1])


def test_archive_rejects_forecast_issued_after_origin(tmp_path: Path) -> None:
    data = make_data(tmp_path / "knowair")
    archive_path = tmp_path / "leaky_forecast.npz"
    write_archive(archive_path, data, issue_after_origin=True)
    archive = ForecastArchive.load(archive_path)
    with pytest.raises(ValueError, match="future-information leakage"):
        archive.validate_for_validation(data, np.array([120, 121]), horizon=24)
