#!/usr/bin/env python3
"""Evaluate frozen Direct and TSR checkpoints on an external panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common_local.data import CommonLocalWindowDataset, load_standard_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.metrics import validation_report
from common_local.model import CommonLocalForecaster
from common_local.paired_statistics import origin_mae, summarize_paired_errors
from common_local.train import choose_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, panel, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "common_local")
    if architecture == "common_local":
        model = CommonLocalForecaster(
            stations=len(panel.stations), **checkpoint.get("config", {})
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    elif architecture == "transport_source_recurrent":
        model = TransportSourceRecurrentForecaster(
            stations=len(panel.stations), coordinates=panel.coordinates,
            **checkpoint.get("config", {}),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    model.eval()
    return model, architecture


def predict(model, loader, panel, device: torch.device):
    predictions, truths, validity, starts = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            value = model(device_batch)["prediction"].cpu().numpy()
            predictions.append(value * panel.std[0] + panel.mean[0])
            truths.append(batch["y"].numpy() * panel.std[0] + panel.mean[0])
            validity.append(batch["y_valid"].numpy())
            starts.append(batch["forecast_start"].numpy())
    return tuple(np.concatenate(values) for values in (predictions, truths, validity, starts))


def mean_metrics(rows: list[dict], key: str) -> dict:
    return {
        metric: {
            "mean": float(np.mean([row[key][metric] for row in rows])),
            "std": float(np.std([row[key][metric] for row in rows])),
        }
        for metric in ("mae", "rmse", "smape")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--panel", default="data/processed/beijing_multisite_3h.npz")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--direct-pattern", default="artifacts/external_beijing/direct/seed_{seed}/best_model.pt"
    )
    parser.add_argument(
        "--tsr-pattern", default="artifacts/external_beijing/tsr_primary/seed_{seed}/best_model.pt"
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_812)
    parser.add_argument("--block-lengths", nargs="+", type=int, default=[56, 112, 224])
    parser.add_argument("--output", required=True)
    parser.add_argument("--series-output")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("Refusing external test access without --unlock-test")

    root = Path(args.root).resolve()
    panel_path = root / args.panel
    panel = load_standard_panel(panel_path, expected_stations=12)
    dataset = CommonLocalWindowDataset(panel, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = choose_device(args.device)
    results, errors, reference_starts, truth_reference = {}, {}, None, None
    checkpoint_rows = []
    for label, pattern in (("direct", args.direct_pattern), ("tsr", args.tsr_pattern)):
        results[label], errors[label] = [], []
        for seed in args.seeds:
            checkpoint_path = root / pattern.format(seed=seed)
            model, architecture = load_model(checkpoint_path, panel, device)
            prediction, truth, valid, starts = predict(model, loader, panel, device)
            if reference_starts is None:
                reference_starts, truth_reference = starts, truth
            elif not np.array_equal(reference_starts, starts) or not np.array_equal(truth_reference, truth):
                raise ValueError("External forecast origins or targets are not exactly aligned")
            metrics = validation_report(
                prediction, truth, cadence_hours=panel.cadence_hours, valid_mask=valid
            )
            results[label].append(metrics)
            errors[label].append(origin_mae(prediction, truth, valid))
            checkpoint_rows.append({
                "label": label, "seed": seed, "architecture": architecture,
                "path": str(checkpoint_path.relative_to(root)),
                "sha256": sha256(checkpoint_path),
                "overall": metrics["overall_1_72h"],
            })

    direct_error = np.stack(errors["direct"])
    tsr_error = np.stack(errors["tsr"])
    paired = summarize_paired_errors(
        direct_error.mean(0), tsr_error.mean(0),
        block_lengths=tuple(args.block_lengths), replicates=args.replicates,
        seed=args.bootstrap_seed,
    )
    key = "overall_1_72h"
    payload = {
        "dataset": "UCI Beijing Multi-Site Air Quality, 3-hour panel",
        "dataset_doi": "10.24432/C5RK5G",
        "panel": str(panel_path.relative_to(root)),
        "panel_sha256": sha256(panel_path),
        "split": args.split,
        "test_accessed": args.split == "test",
        "information_set": "72-hour PM2.5 history and realized target-period meteorology",
        "history_hours": 72,
        "horizon_hours": 72,
        "seeds": args.seeds,
        "direct": mean_metrics(results["direct"], key),
        "tsr": mean_metrics(results["tsr"], key),
        "paired_direct_minus_tsr": paired,
        "bootstrap": {
            "method": "circular moving-block percentile bootstrap",
            "replicates": args.replicates,
            "seed": args.bootstrap_seed,
            "block_lengths_origins": args.block_lengths,
            "block_lengths_hours": [length * panel.cadence_hours for length in args.block_lengths],
        },
        "checkpoints": checkpoint_rows,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    if args.series_output:
        series = root / args.series_output
        series.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            series, forecast_start=reference_starts,
            direct_origin_mae_by_seed=direct_error,
            tsr_origin_mae_by_seed=tsr_error,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
