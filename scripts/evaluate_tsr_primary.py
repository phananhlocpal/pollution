#!/usr/bin/env python3
"""Evaluate the frozen primary TSR family and its uniform ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.metrics import validation_report
from common_local.train import choose_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(checkpoint_path: Path, root: Path, panel, loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = TransportSourceRecurrentForecaster(
        root / "data/benchmarks/knowair/city.txt",
        stations=len(panel.stations),
        **config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            value = model(device_batch)["prediction"].cpu().numpy()
            predictions.append(value * panel.std[0] + panel.mean[0])
            truths.append(batch["y"].numpy() * panel.std[0] + panel.mean[0])
    return np.concatenate(predictions), np.concatenate(truths), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--checkpoint-pattern",
        default="paper/checkpoints/tsr_primary/seed_{seed}.pt",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test access without --allow-test")

    root = Path(args.root).resolve()
    panel = load_panel(root)
    dataset = CommonLocalWindowDataset(panel, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = choose_device(args.device)
    seed_predictions, seed_reports, checkpoints = [], [], []
    truth_reference = None
    for seed in args.seeds:
        checkpoint_path = root / args.checkpoint_pattern.format(seed=seed)
        prediction, truth, checkpoint = predict(
            checkpoint_path, root, panel, loader, device
        )
        if truth_reference is None:
            truth_reference = truth
        elif not np.array_equal(truth_reference, truth):
            raise ValueError("Targets are not aligned across seeds")
        report = validation_report(prediction, truth, panel.cadence_hours)
        seed_predictions.append(prediction)
        seed_reports.append(report)
        checkpoints.append({
            "seed": seed,
            "path": str(checkpoint_path.relative_to(root)),
            "sha256": sha256(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "metrics": report["overall_1_72h"],
        })

    ensemble = np.mean(np.stack(seed_predictions), axis=0)
    ensemble_report = validation_report(ensemble, truth_reference, panel.cadence_hours)
    overall_key = "overall_1_72h"
    payload = {
        "method": "Transport-Source Recurrent Operator",
        "split": args.split,
        "test_accessed": args.split == "test",
        "selection": "architecture and checkpoints selected on validation before this evaluation",
        "test_history": "KnowAir test had previously been viewed at project level during model development",
        "information_set": "72-hour PM2.5 history and realized target-period meteorology",
        "seeds": args.seeds,
        "single_models": {
            "mae_mean": float(np.mean([
                report[overall_key]["mae"] for report in seed_reports
            ])),
            "mae_std": float(np.std([
                report[overall_key]["mae"] for report in seed_reports
            ])),
            "metrics_by_seed": checkpoints,
        },
        "uniform_ensemble": ensemble_report[overall_key],
        "uniform_ensemble_day_mae": [
            ensemble_report[key]["mae"]
            for key in ("day1_1_24h", "day2_25_48h", "day3_49_72h")
        ],
        "checkpoints": checkpoints,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
