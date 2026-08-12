#!/usr/bin/env python3
"""Paired block-bootstrap comparison of two aligned checkpoint families."""

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
from common_local.model import CommonLocalForecaster
from common_local.paired_statistics import origin_mae, summarize_paired_errors
from common_local.train import choose_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, root: Path, panel, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "common_local")
    if architecture == "transport_source_recurrent":
        model = TransportSourceRecurrentForecaster(
            root / "data/benchmarks/knowair/city.txt",
            stations=len(panel.stations), **checkpoint.get("config", {}),
        ).to(device)
        incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
        missing = [key for key in incompatible.missing_keys if key != "station_threshold"]
        if missing or incompatible.unexpected_keys:
            raise ValueError(
                f"Incompatible TSR checkpoint {checkpoint_path}: "
                f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
            )
    elif architecture == "common_local":
        model = CommonLocalForecaster(
            stations=len(panel.stations), **checkpoint.get("config", {})
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError(f"Unsupported architecture {architecture!r} in {checkpoint_path}")
    model.eval()
    return model, architecture


def evaluate_origins(model, loader, panel, device: torch.device):
    errors, starts = [], []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model(device_batch)["prediction"].detach().cpu().numpy()
            prediction = prediction * panel.std[0] + panel.mean[0]
            truth = batch["y"].numpy() * panel.std[0] + panel.mean[0]
            errors.append(origin_mae(prediction, truth, batch.get("y_valid")))
            starts.append(batch["forecast_start"].numpy())
    return np.concatenate(errors), np.concatenate(starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--reference-pattern",
        default="paper/checkpoints/core_meteo_lagged/seed_{seed}.pt",
    )
    parser.add_argument(
        "--proposed-pattern", default="paper/checkpoints/tsr_primary/seed_{seed}.pt"
    )
    parser.add_argument("--reference-label", default="TSR plus fixed one-step delay")
    parser.add_argument("--proposed-label", default="primary TSR")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_812)
    parser.add_argument("--block-lengths", nargs="+", type=int, default=[56, 112, 224])
    parser.add_argument(
        "--output", default="paper/artifacts/paired_block_bootstrap_lagged_vs_no_lag.json"
    )
    parser.add_argument(
        "--series-output", default="paper/artifacts/paired_origin_errors_lagged_vs_no_lag.npz"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    panel = load_panel(root)
    dataset = CommonLocalWindowDataset(panel, "val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = choose_device(args.device)
    reference_by_seed, proposed_by_seed, checkpoint_manifest = [], [], []
    reference_starts = None
    for seed in args.seeds:
        pair = {}
        for label, pattern in (
            ("reference", args.reference_pattern),
            ("proposed", args.proposed_pattern),
        ):
            checkpoint_path = root / pattern.format(seed=seed)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model, architecture = load_model(checkpoint_path, root, panel, device)
            error, starts = evaluate_origins(model, loader, panel, device)
            if reference_starts is None:
                reference_starts = starts
            elif not np.array_equal(reference_starts, starts):
                raise ValueError("Forecast origins are not exactly aligned")
            pair[label] = error
            checkpoint_manifest.append({
                "seed": seed,
                "label": label,
                "architecture": architecture,
                "path": str(checkpoint_path.relative_to(root)),
                "sha256": sha256(checkpoint_path),
                "origin_mae": float(error.mean()),
            })
        reference_by_seed.append(pair["reference"])
        proposed_by_seed.append(pair["proposed"])

    reference = np.stack(reference_by_seed)
    proposed = np.stack(proposed_by_seed)
    mean_reference = reference.mean(axis=0)
    mean_proposed = proposed.mean(axis=0)
    summary = summarize_paired_errors(
        mean_reference,
        mean_proposed,
        block_lengths=tuple(args.block_lengths),
        replicates=args.replicates,
        seed=args.bootstrap_seed,
    )
    summary.update({
        "comparison": f"{args.reference_label} minus {args.proposed_label}",
        "reference_label": args.reference_label,
        "proposed_label": args.proposed_label,
        "split": "KnowAir validation (chronological 50--75%)",
        "test_accessed": False,
        "cadence_hours": panel.cadence_hours,
        "history_steps": 24,
        "horizon_steps": 24,
        "aggregation": (
            "MAE over valid horizon-station targets within each origin; mean over "
            "three seed-specific error series before temporal resampling"
        ),
        "bootstrap": "circular moving-block percentile bootstrap",
        "replicates": args.replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "block_lengths_origins": args.block_lengths,
        "block_lengths_hours": [value * panel.cadence_hours for value in args.block_lengths],
        "checkpoints": checkpoint_manifest,
    })
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    series_output = root / args.series_output
    series_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        series_output,
        forecast_start=reference_starts,
        reference_origin_mae_by_seed=reference,
        proposed_origin_mae_by_seed=proposed,
        reference_origin_mae=mean_reference,
        proposed_origin_mae=mean_proposed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
