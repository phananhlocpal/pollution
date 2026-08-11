"""Evaluate frozen corrected China-AQI 96->24 checkpoints once explicitly unlocked."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common_local.data import GAGNNAirDDEWindowDataset, load_gagnn_metadata
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.metrics import validation_report
from common_local.train import _loader, choose_device, move_batch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.allow_test:
        raise SystemExit("Refusing to open corrected China-AQI test without --allow-test")
    root = Path(args.root)
    manifest_path = root / "frozen/china_aqi_96x24_latent_v2/MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit("Freeze corrected checkpoints first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_accessed") is not False:
        raise SystemExit("Manifest is not in a sealed pre-test state")
    output = root / "artifacts/china_aqi_96x24_latent_v2/TEST_RESULT.json"
    if output.exists():
        raise SystemExit(f"Refusing to repeat recorded test evaluation: {output}")

    data_root = root / "data/benchmarks/china_aqi_gagnn"
    panel = load_gagnn_metadata(data_root, "96x24")
    dataset = GAGNNAirDDEWindowDataset(data_root, "test", panel)
    device = choose_device(args.device)
    rows, prediction_arrays, truth_physical = [], [], None
    for seed in manifest["seeds"]:
        checkpoint_path = root / f"artifacts/china_aqi_96x24_latent_v2/seed_{seed}/best_model.pt"
        if sha256(checkpoint_path) != manifest["checkpoint_sha256"][str(seed)]:
            raise SystemExit(f"Checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = TransportSourceRecurrentForecaster(
            stations=209, coordinates=panel.coordinates, **checkpoint["config"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        predictions, truths = [], []
        model.eval()
        with torch.no_grad():
            for batch in _loader(dataset, args.batch_size, seed, False):
                batch = move_batch(batch, device)
                predictions.append(model(batch)["prediction"].cpu().numpy())
                truths.append(batch["y"].cpu().numpy())
        prediction = np.concatenate(predictions) * panel.std[0] + panel.mean[0]
        truth = np.concatenate(truths) * panel.std[0] + panel.mean[0]
        if truth_physical is not None and not np.array_equal(truth_physical, truth):
            raise SystemExit("Truth mismatch across seed evaluation")
        truth_physical = truth
        prediction_arrays.append(prediction)
        metrics = validation_report(prediction, truth, cadence_hours=1)["overall_1_24h"]
        rows.append({"seed": seed, "metrics": metrics})
    metric_names = ("mae", "rmse", "mape")
    ensemble_metrics = validation_report(
        np.mean(np.stack(prediction_arrays), axis=0), truth_physical, cadence_hours=1
    )["overall_1_24h"]
    summary = {
        "dataset": "China-AQI official GAGNN release, reconstructed 96h->24h",
        "information_set": "historical AQI and meteorology only",
        "single_models": rows,
        "three_seed_mean": {
            name: float(np.mean([row["metrics"][name] for row in rows]))
            for name in metric_names
        },
        "three_seed_std": {
            name: float(np.std([row["metrics"][name] for row in rows]))
            for name in metric_names
        },
        "uniform_mean_ensemble": {name: ensemble_metrics[name] for name in metric_names},
        "published_airdde_same_protocol": {"mae": 17.03, "rmse": 29.91, "mape": 30.82},
        "disclosure": manifest["disclosure"],
        "test_accessed": True,
    }
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
