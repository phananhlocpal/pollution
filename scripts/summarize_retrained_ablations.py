"""Summarize independently retrained mechanism ablations on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = {
    "full_recurrent": "artifacts/retrained_ablation_full",
    "no_transport": "artifacts/retrained_ablation_no_transport",
    "no_source_sink": "artifacts/retrained_ablation_no_source",
    "matched_direct": "artifacts/common_local_matched",
}
PERIODS = ("day1_1_24h", "day2_25_48h", "day3_49_72h", "overall_1_72h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/retrained_ablation/summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    variants = {}
    for name, relative in VARIANTS.items():
        rows = []
        for seed in (42, 43, 44):
            path = root / relative / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            metrics = payload["validation"]["metrics"]
            rows.append({
                "seed": seed,
                "parameters": payload["parameter_count"],
                "best_epoch": payload["best_epoch"],
                **{period: metrics[period]["mae"] for period in PERIODS},
            })
        if len(rows) != 3:
            raise SystemExit(f"{name}: expected seeds 42/43/44, found {[r['seed'] for r in rows]}")
        variants[name] = {
            "seeds": rows,
            "parameter_count": rows[0]["parameters"],
            "mean_mae": {period: float(np.mean([r[period] for r in rows])) for period in PERIODS},
            "std_mae": {period: float(np.std([r[period] for r in rows], ddof=1)) for period in PERIODS},
            "test_accessed": False,
        }
    full = variants["full_recurrent"]["mean_mae"]["overall_1_72h"]
    for payload in variants.values():
        payload["delta_overall_mae_vs_full"] = payload["mean_mae"]["overall_1_72h"] - full
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "independently_retrained_mechanism_ablation",
        "selection_split": "validation",
        "seeds": [42, 43, 44],
        "variants": variants,
        "note": "Deltas are retraining comparisons, distinct from frozen-checkpoint knockouts.",
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
