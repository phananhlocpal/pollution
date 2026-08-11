"""Model-independent evaluator for aligned pollution forecast bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common_local.metrics import validation_report


def load_bundle(directory: str | Path):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    prediction = np.load(directory / "prediction.npy", mmap_mode="r")
    truth = np.load(directory / "truth.npy", mmap_mode="r")
    starts = np.load(directory / "forecast_start.npy")
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction/truth must have the same [origin,horizon,node] shape")
    if len(starts) != len(prediction):
        raise ValueError("forecast_start is not aligned with predictions")
    return manifest, prediction, truth, starts


def origin_mae(prediction, truth):
    valid = truth >= 1e-4
    absolute = np.where(valid, np.abs(prediction - truth), np.nan)
    return np.nanmean(absolute, axis=(1, 2))


def paired_block_interval(loss_a, loss_b, block_length=24, repetitions=2000, seed=2026):
    """Moving-block bootstrap CI for mean paired loss A-B over forecast origins."""
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(difference)
    if n < block_length:
        raise ValueError(f"Need at least {block_length} origins for block bootstrap")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block_length + 1)
    draws = np.empty(repetitions)
    blocks = int(np.ceil(n / block_length))
    for index in range(repetitions):
        selected = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([difference[s:s + block_length] for s in selected])[:n]
        draws[index] = sample.mean()
    return {
        "mean_delta_mae_a_minus_b": float(difference.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (.025, .975))],
        "block_length_origins": block_length,
        "bootstrap_repetitions": repetitions,
    }


def evaluate(directory, persistence=True):
    manifest, prediction, truth, starts = load_bundle(directory)
    report = validation_report(prediction, truth)
    if persistence:
        if "persistence.npy" not in {path.name for path in Path(directory).iterdir()}:
            raise FileNotFoundError("Bundle has no persistence.npy")
        persistence_prediction = np.load(Path(directory) / "persistence.npy", mmap_mode="r")
        persistence_report = validation_report(persistence_prediction, truth)
        report["persistence"] = persistence_report
        report["skill_mae"] = {
            period: 1 - report[period]["mae"] / persistence_report[period]["mae"]
            for period in ("day1_1_24h", "day2_25_48h", "day3_49_72h", "overall_1_72h")
        }
    return {"manifest": manifest, "origins": len(starts), "metrics": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Directory containing prediction/truth .npy files")
    parser.add_argument("--compare", help="Second aligned prediction bundle")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=24)
    args = parser.parse_args()
    result = evaluate(args.bundle)
    if args.compare:
        manifest_a, pred_a, truth_a, starts_a = load_bundle(args.bundle)
        manifest_b, pred_b, truth_b, starts_b = load_bundle(args.compare)
        if pred_a.shape != pred_b.shape or not np.array_equal(starts_a, starts_b):
            raise ValueError("Compared bundles do not contain identical forecast origins")
        if not np.allclose(truth_a, truth_b, equal_nan=True):
            raise ValueError("Compared bundles do not contain identical truth tensors")
        result["comparison"] = paired_block_interval(
            origin_mae(pred_a, truth_a), origin_mae(pred_b, truth_b),
            args.block_length, args.bootstrap_repetitions,
        )
        result["comparison"]["a"] = manifest_a.get("model")
        result["comparison"]["b"] = manifest_b.get("model")
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

