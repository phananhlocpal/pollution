"""Freeze China-AQI checkpoints and validation decisions before opening test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/china_aqi_history_learned"
OUTPUT = ROOT / "frozen/china_aqi/MANIFEST.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing freeze manifest: {OUTPUT}")
    checkpoints, validation = {}, {}
    for seed in (42, 43, 44):
        folder = SOURCE / f"seed_{seed}"
        checkpoint = folder / "best_model.pt"
        metrics = json.loads((folder / "metrics.json").read_text())
        checkpoints[str(seed)] = sha256(checkpoint)
        validation[str(seed)] = metrics["best_validation_mae"]
    payload = {
        "status": "frozen_before_china_aqi_test",
        "architecture": "transport_source_recurrent",
        "future_weather_mode": "learned_causal_history_only",
        "dataset_source": "official GAGNN release",
        "gagnn_commit": "509ac7d6eb55914979fc45f6d23e967021cfd270",
        "dataset_archive_sha256": "4decea9aaafcd60e08e495083d708da238ab84ba0b507d42fe692c7405b52fd8",
        "locations": 209,
        "history": 24,
        "horizon": 6,
        "cadence_hours": 1,
        "official_split": [0.7, 0.1, 0.2],
        "seeds": [42, 43, 44],
        "checkpoint_sha256": checkpoints,
        "validation_mae_by_seed": validation,
        "test_reporting": {
            "primary": "single_model_three_seed_mean_and_std",
            "secondary": "uniform_three_seed_mean_ensemble",
            "convex_ensemble": False
        },
        "test_accessed": False
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
