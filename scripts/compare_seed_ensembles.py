"""Seed-aware paired comparison for aligned multi-seed prediction bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarking.evaluator import load_bundle, origin_mae
from common_local.metrics import validation_report


def load_group(paths):
    loaded = [load_bundle(path) for path in paths]
    truth, starts = loaded[0][2], loaded[0][3]
    for _, _, candidate_truth, candidate_starts in loaded[1:]:
        if not np.array_equal(starts, candidate_starts):
            raise ValueError("Seed bundles have different forecast origins")
        if not np.allclose(truth, candidate_truth, equal_nan=True):
            raise ValueError("Seed bundles have different truth tensors")
    losses = np.stack([origin_mae(item[1], truth) for item in loaded])
    metrics = [validation_report(item[1], truth)["overall_1_72h"] for item in loaded]
    return loaded, truth, starts, losses, metrics


def hierarchical_interval(loss_a, loss_b, block_length, repetitions, seed=2026):
    rng = np.random.default_rng(seed)
    origins = loss_a.shape[1]
    starts = np.arange(origins - block_length + 1)
    blocks = int(np.ceil(origins / block_length))
    draws = np.empty(repetitions)
    for index in range(repetitions):
        seed_a = rng.integers(0, len(loss_a), len(loss_a))
        seed_b = rng.integers(0, len(loss_b), len(loss_b))
        origin_blocks = rng.choice(starts, blocks, replace=True)
        origin_index = np.concatenate([
            np.arange(start, start + block_length) for start in origin_blocks
        ])[:origins]
        draws[index] = (
            loss_a[seed_a][:, origin_index].mean()
            - loss_b[seed_b][:, origin_index].mean()
        )
    return {
        "mean_delta_mae_a_minus_b": float(loss_a.mean() - loss_b.mean()),
        "hierarchical_ci95": [float(value) for value in np.quantile(draws, (.025, .975))],
        "block_length_origins": block_length,
        "bootstrap_repetitions": repetitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    parser.add_argument("--block-length", type=int, default=24)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output")
    args = parser.parse_args()
    loaded_a, truth_a, starts_a, loss_a, metrics_a = load_group(args.a)
    loaded_b, truth_b, starts_b, loss_b, metrics_b = load_group(args.b)
    if not np.array_equal(starts_a, starts_b) or not np.allclose(truth_a, truth_b):
        raise ValueError("Model groups are not paired on identical samples")
    result = {
        "a": [item[0].get("model") for item in loaded_a],
        "b": [item[0].get("model") for item in loaded_b],
        "a_seed_mae": [row["mae"] for row in metrics_a],
        "b_seed_mae": [row["mae"] for row in metrics_b],
        "a_mean_std_mae": [float(loss_a.mean()), float(np.std([row["mae"] for row in metrics_a], ddof=1))],
        "b_mean_std_mae": [float(loss_b.mean()), float(np.std([row["mae"] for row in metrics_b], ddof=1))],
        "comparison": hierarchical_interval(
            loss_a, loss_b, args.block_length, args.bootstrap_repetitions
        ),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
