"""Freeze corrected China-AQI 96->24 V2 checkpoints before test evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/china_aqi_96x24_latent_v2"
OUTPUT = ROOT / "frozen/china_aqi_96x24_latent_v2/MANIFEST.json"


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
        if metrics.get("gagnn_protocol") != "96x24":
            raise SystemExit(f"Seed {seed} was not trained with corrected 96x24 protocol")
        config = metrics["config"]
        if config.get("future_weather_mode") != "latent" or config.get("use_lagged_transport"):
            raise SystemExit(f"Seed {seed} is not the frozen history-only no-lag V2")
        checkpoints[str(seed)] = sha256(checkpoint)
        validation[str(seed)] = metrics["best_validation_mae"]
    audit_path = ROOT / "artifacts/external_replication/china_aqi_reconstruction.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("reconstructable"):
        raise SystemExit("Exact GAGNN reconstruction audit did not pass")
    payload = {
        "status": "frozen_before_corrected_china_aqi_test",
        "architecture": "latent_forcing_transport_source_recurrent_v2",
        "information_set": "historical AQI and historical meteorology only",
        "explicit_lagged_transport": False,
        "dataset_source": "official GAGNN release reconstructed split-locally",
        "gagnn_commit": "509ac7d6eb55914979fc45f6d23e967021cfd270",
        "dataset_archive_sha256": "4decea9aaafcd60e08e495083d708da238ab84ba0b507d42fe692c7405b52fd8",
        "locations": 209,
        "history": 96,
        "horizon": 24,
        "cadence_hours": 1,
        "released_split_policy": "train/val/test reconstructed independently",
        "seeds": [42, 43, 44],
        "checkpoint_sha256": checkpoints,
        "validation_mae_by_seed": validation,
        "test_reporting": {
            "primary": "single_model_three_seed_mean_and_std",
            "secondary": "uniform_three_seed_mean_ensemble",
            "convex_ensemble": False,
        },
        "disclosure": (
            "The same released test period was previously viewed under the different "
            "24h->6h task; this is a protocol correction, not a pristine lockbox."
        ),
        "test_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
