"""Compare residual structure before and after recurrent state evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.evaluator import evaluate


def averaged_table(root: Path, seeds, filename, keys):
    frames = []
    for seed in seeds:
        frame = pd.read_csv(root / f"seed_{seed}" / filename)
        frame["seed"] = seed
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    numeric = [column for column in table.columns if column not in {*keys, "seed"}]
    return table.groupby(list(keys), as_index=False)[numeric].mean()


def records_at(table, horizon):
    return table.loc[table["horizon_hours"] == horizon].to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", default="artifacts/residual_generation_comparison/summary.json")
    args = parser.parse_args()
    root = Path(args.root); seeds = tuple(args.seeds)
    before_root = root / "artifacts/residual_probe"
    after_root = root / "artifacts/residual_probe_recurrent_core_meteo"
    before_event = averaged_table(before_root, seeds, "residual_by_event_phase.csv", ("horizon_hours", "event_phase"))
    after_event = averaged_table(after_root, seeds, "residual_by_event_phase.csv", ("horizon_hours", "event_phase"))
    before_season = averaged_table(before_root, seeds, "residual_by_season.csv", ("horizon_hours", "season"))
    after_season = averaged_table(after_root, seeds, "residual_by_season.csv", ("horizon_hours", "season"))
    before_probe = averaged_table(before_root, seeds, "chronological_ridge_probes.csv", ("horizon_hours", "signal"))
    after_probe = averaged_table(after_root, seeds, "chronological_ridge_probes.csv", ("horizon_hours", "signal"))

    horizons = [3, 6, 12, 24, 36, 48, 72]
    before_horizon, after_horizon = [], []
    for seed in seeds:
        before = evaluate(root / f"artifacts/predictions/common_local_seed{seed}_val")["metrics"]
        after = evaluate(root / f"artifacts/predictions/transport_source_recurrent_strict_seed{seed}_val")["metrics"]
        before_horizon.append(before["mae_by_horizon"])
        after_horizon.append(after["mae_by_horizon"])
    before_horizon = np.mean(before_horizon, axis=0)
    after_horizon = np.mean(after_horizon, axis=0)

    probe_signals = (
        "wind_aligned_innovation", "boundary_layer_height", "ventilation",
    )
    probe = []
    for horizon in (3, 24, 72):
        for signal in probe_signals:
            old = before_probe[(before_probe.horizon_hours == horizon) & (before_probe.signal == signal)]
            new = after_probe[(after_probe.horizon_hours == horizon) & (after_probe.signal == signal)]
            if len(old) and len(new):
                probe.append({
                    "horizon_hours": horizon, "signal": signal,
                    "before_incremental_delta_mae": float(old.iloc[0].incremental_delta_vs_baseline_probe),
                    "after_incremental_delta_mae": float(new.iloc[0].incremental_delta_vs_baseline_probe),
                })
    result = {
        "models": {"before": "common_local", "after": "core-meteorology recurrent"},
        "split": "validation", "seeds": list(seeds), "test_accessed": False,
        "horizon_mae": [{
            "horizon_hours": horizon,
            "before": float(before_horizon[horizon // 3 - 1]),
            "after": float(after_horizon[horizon // 3 - 1]),
            "reduction": float(before_horizon[horizon // 3 - 1] - after_horizon[horizon // 3 - 1]),
        } for horizon in horizons],
        "event_phase_72h": {
            "before": records_at(before_event, 72), "after": records_at(after_event, 72),
        },
        "season_72h": {
            "before": records_at(before_season, 72), "after": records_at(after_season, 72),
        },
        "residual_probe_incremental_gain": probe,
        "interpretation_rule": "Negative incremental delta means the residual signal remains correctable on the purged probe split.",
    }
    output = root / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
