"""Build mean, median, and validation-fitted convex ensembles across three seeds."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from benchmarking.evaluator import evaluate, load_bundle


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/predictions"
SEEDS = (42, 43, 44)


def sources(split, prefix="common_local_wind_meteo", seeds=SEEDS):
    return [PREDICTIONS / f"{prefix}_seed{seed}_{split}" for seed in seeds]


def fit_convex_weights(prefix="common_local_wind_meteo", seeds=SEEDS, sample_points=1_000_000):
    bundles = [load_bundle(path) for path in sources("val", prefix, seeds)]
    truth = bundles[0][2]
    total = truth.size
    indices = np.linspace(0, total - 1, min(sample_points, total), dtype=np.int64)
    target = np.asarray(truth).reshape(-1)[indices]
    predictions = np.stack([np.asarray(bundle[1]).reshape(-1)[indices] for bundle in bundles], axis=1)
    valid = target >= 1e-4; target = target[valid]; predictions = predictions[valid]

    def objective(weights):
        return np.mean(np.abs(predictions @ weights - target))

    result = minimize(
        objective, np.full(len(seeds), 1 / len(seeds)), method="SLSQP",
        bounds=[(0, 1)] * len(seeds),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x / result.x.sum(), float(result.fun), int(valid.sum())


def hardlink(source, destination):
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        import shutil
        shutil.copy2(source, destination)


def build(split, method, weights=None, chunk=64,
          prefix="common_local_wind_meteo", seeds=SEEDS):
    loaded = [load_bundle(path) for path in sources(split, prefix, seeds)]
    manifests = [item[0] for item in loaded]
    predictions = [item[1] for item in loaded]
    truth, starts = loaded[0][2], loaded[0][3]
    for _, _, candidate_truth, candidate_starts in loaded[1:]:
        if not np.array_equal(starts, candidate_starts) or not np.allclose(truth, candidate_truth):
            raise ValueError("Seed bundles are not aligned")
    output = PREDICTIONS / f"{prefix}_{method}_ensemble_{split}"
    output.mkdir(parents=True, exist_ok=True)
    destination = np.lib.format.open_memmap(
        output / "prediction.npy", "w+", dtype="float32", shape=predictions[0].shape
    )
    for left in range(0, len(destination), chunk):
        right = min(left + chunk, len(destination))
        stacked = np.stack([np.asarray(value[left:right]) for value in predictions])
        if method == "mean":
            destination[left:right] = stacked.mean(0)
        elif method == "median":
            destination[left:right] = np.median(stacked, axis=0)
        elif method == "convex":
            destination[left:right] = np.tensordot(weights, stacked, axes=(0, 0))
        else:
            raise ValueError(method)
    destination.flush()
    first = sources(split, prefix, seeds)[0]
    for filename in ("truth.npy", "persistence.npy", "forecast_start.npy"):
        hardlink(first / filename, output / filename)
    manifest = {
        "model": f"{prefix}:{method}_ensemble", "split": split,
        "seeds": list(seeds), "weights": None if weights is None else weights.tolist(),
        "shape": list(destination.shape), "dataset_sha256": manifests[0]["dataset_sha256"],
        "station_order_sha256": manifests[0]["station_order_sha256"],
        "source_checkpoint_sha256": [item["checkpoint_sha256"] for item in manifests],
        "weight_selection_split": "validation" if method == "convex" else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = evaluate(output)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="common_local_wind_meteo")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--summary-output", default="artifacts/ensemble/summary.json")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    weights, sampled_mae, fitted_points = fit_convex_weights(args.prefix, seeds)
    summary = {"prefix": args.prefix, "seeds": list(seeds),
               "convex_weights": weights.tolist(), "fit_sample_mae": sampled_mae,
               "fit_valid_points": fitted_points, "results": {}}
    for split in args.splits:
        for method in ("mean", "median", "convex"):
            report = build(split, method, weights, prefix=args.prefix, seeds=seeds)
            summary["results"][f"{method}_{split}"] = report["metrics"]["overall_1_72h"]
    output = ROOT / args.summary_output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
