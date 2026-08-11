"""Aggregate residual screening results across common_local seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/residual_probe")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", default="artifacts/residual_probe/aggregate")
    args = parser.parse_args()
    probes, correlations = [], []
    for seed in args.seeds:
        directory = Path(args.root) / f"seed_{seed}"
        probe = pd.read_csv(directory / "chronological_ridge_probes.csv")
        probe["seed"] = seed; probes.append(probe)
        correlation = pd.read_csv(directory / "residual_signal_correlations.csv")
        correlation["seed"] = seed; correlations.append(correlation)
    probes = pd.concat(probes, ignore_index=True)
    correlations = pd.concat(correlations, ignore_index=True)
    signals = probes[probes.signal != "baseline_current_pm+prediction"].copy()
    by_horizon = signals.groupby(["horizon_hours", "signal"]).agg(
        mean_incremental_delta=("incremental_delta_vs_baseline_probe", "mean"),
        std_incremental_delta=("incremental_delta_vs_baseline_probe", "std"),
        improving_seeds=("incremental_delta_vs_baseline_probe", lambda x: int((x < 0).sum())),
        mean_corrected_mae=("corrected_mae", "mean"),
    ).reset_index()
    ranking = signals.groupby("signal").agg(
        mean_incremental_delta=("incremental_delta_vs_baseline_probe", "mean"),
        std_incremental_delta=("incremental_delta_vs_baseline_probe", "std"),
        improving_seed_horizons=("incremental_delta_vs_baseline_probe", lambda x: int((x < 0).sum())),
        comparisons=("incremental_delta_vs_baseline_probe", "size"),
    ).reset_index().sort_values("mean_incremental_delta")
    correlation_summary = correlations.groupby(["horizon_hours", "signal"]).agg(
        mean_residual_correlation=("residual_correlation", "mean"),
        std_residual_correlation=("residual_correlation", "std"),
    ).reset_index()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    by_horizon.to_csv(output / "probe_by_horizon.csv", index=False)
    ranking.to_csv(output / "candidate_ranking.csv", index=False)
    correlation_summary.to_csv(output / "correlation_by_horizon.csv", index=False)
    payload = {
        "seeds": args.seeds,
        "selection_rule": "Advance only signals improving all 3 seeds at a relevant horizon; values are screening, not final inference.",
        "best_mean_signal": ranking.iloc[0].to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2))
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()

