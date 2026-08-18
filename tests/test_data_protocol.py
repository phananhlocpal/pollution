from __future__ import annotations

from pathlib import Path

import numpy as np

from shift_pm import KnowAirDataModule
from shift_pm.data import RAW_FEATURES


def _synthetic_knowair(root: Path, length: int = 240, stations: int = 4) -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(length, stations, len(RAW_FEATURES))).astype(np.float64)
    pm = 40 + rng.uniform(0, 20, size=(length, stations))
    pm[150, 1] = 0.0
    raw[..., RAW_FEATURES.index("pm25")] = pm
    weather_names = ("temperature", "pressure", "rh950", "u100", "v100")
    for feature_index, name in enumerate(weather_names):
        values = 10.0 + feature_index + rng.normal(0, 0.1, size=(length, stations))
        # Large post-train shift must not affect fitted normalization.
        values[120:] += 1000.0
        raw[..., RAW_FEATURES.index(name)] = values
    np.save(root / "KnowAir.npy", raw)
    with (root / "city.txt").open("w", encoding="utf-8") as handle:
        for station in range(stations):
            handle.write(f"{station} S{station} {100 + station * .5} {30 + station * .2}\n")
    return root


def test_router_boundaries_and_train_only_pm_transform(tmp_path: Path) -> None:
    data = KnowAirDataModule(_synthetic_knowair(tmp_path / "knowair"))

    assert data.boundaries == {
        "train": (0, 120), "val": (120, 180), "test": (180, 240)
    }
    assert data.pm_anomaly.shape == (240, 4)
    assert np.isfinite(data.pm_anomaly).all()
    assert data.transform.log_climatology.shape == (12, 8, 4)
    assert data.transform.anomaly_scale.shape == (4,)
