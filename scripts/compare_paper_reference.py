"""Compare one frozen bundle with a published scalar reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarking.evaluator import evaluate, load_bundle, origin_mae


def block_mean_interval(loss, block_length, repetitions=5000, seed=2026):
    loss = np.asarray(loss)
    rng = np.random.default_rng(seed)
    starts = np.arange(len(loss) - block_length + 1)
    blocks = int(np.ceil(len(loss) / block_length))
    draws = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.choice(starts, blocks, replace=True)
        sample = np.concatenate([loss[start:start + block_length] for start in selected])
        draws[index] = sample[:len(loss)].mean()
    return [float(value) for value in np.quantile(draws, (.025, .975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--paper-mae", type=float, default=16.92)
    parser.add_argument("--block-lengths", nargs="+", type=int, default=(24, 48, 96))
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/next_generation/paper_comparison.json")
    args = parser.parse_args()
    manifest, prediction, truth, _ = load_bundle(args.bundle)
    report = evaluate(args.bundle)["metrics"]
    loss = origin_mae(prediction, truth)
    model_mae = report["overall_1_72h"]["mae"]
    intervals = {
        str(block): block_mean_interval(loss, block, args.bootstrap_repetitions)
        for block in args.block_lengths
    }
    result = {
        "model": manifest.get("model"), "bundle": args.bundle,
        "paper_reference_mae": args.paper_mae,
        "model_test_mae": model_mae,
        "improvement_mae": args.paper_mae - model_mae,
        "relative_improvement_percent": 100 * (args.paper_mae - model_mae) / args.paper_mae,
        "model_test_rmse": report["overall_1_72h"]["rmse"],
        "model_test_smape_official": report["overall_1_72h"]["smape"],
        "model_test_smape_masked": report["overall_1_72h"]["smape_masked"],
        "test_day_mae": [report[name]["mae"] for name in (
            "day1_1_24h", "day2_25_48h", "day3_49_72h"
        )],
        "origin_block_bootstrap_model_mae_ci95": intervals,
        "paper_reference_outside_model_ci": all(
            args.paper_mae > interval[1] for interval in intervals.values()
        ),
        "note": "The paper exposes a scalar, not paired origin losses; these are one-sample temporal-block intervals for the frozen model.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
