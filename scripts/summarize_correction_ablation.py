"""Create the compact research summary for frozen residual-correction ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "artifacts/correction_ablation"


def main():
    screen = []
    for variant in ("spatial", "wind", "meteo", "regional", "wind_meteo", "wind_spatial"):
        payload = json.loads((ABLATION / variant / "seed_43/metrics.json").read_text())
        screen.append({"variant": variant, "seed": 43, "mae": payload["validation"]["metrics"]["overall_1_72h"]["mae"],
                       "delta_mae": payload["delta_mae"],
                       "trainable_parameters": payload["trainable_parameters"]})
    replication = []
    for seed in (42, 43, 44):
        payload = json.loads((ABLATION / f"wind_meteo/seed_{seed}/metrics.json").read_text())
        metrics = payload["validation"]["metrics"]
        replication.append({
            "seed": seed, "baseline_mae": payload["baseline_validation_mae"],
            "day1_mae": metrics["day1_1_24h"]["mae"],
            "day2_mae": metrics["day2_25_48h"]["mae"],
            "day3_mae": metrics["day3_49_72h"]["mae"],
            "overall_mae": metrics["overall_1_72h"]["mae"], "delta_mae": payload["delta_mae"],
            "trainable_parameters": payload["trainable_parameters"],
            "total_parameters": payload["total_parameters"],
        })
    screen_frame = pd.DataFrame(screen).sort_values("mae")
    replication_frame = pd.DataFrame(replication)
    screen_frame.to_csv(ABLATION / "seed43_screen.csv", index=False)
    replication_frame.to_csv(ABLATION / "wind_meteo_three_seed.csv", index=False)
    summary = {
        "selected_variant": "wind_meteo",
        "selection_reason": "Best seed-43 validation MAE among minimal independent/cumulative corrections.",
        "three_seed_mean": {
            column: float(replication_frame[column].mean())
            for column in ("baseline_mae", "day1_mae", "day2_mae", "day3_mae", "overall_mae", "delta_mae")
        },
        "three_seed_std": {
            column: float(replication_frame[column].std(ddof=0))
            for column in ("overall_mae", "delta_mae")
        },
        "improved_all_seeds": bool((replication_frame.delta_mae < 0).all()),
        "parameter_count": int(replication_frame.total_parameters.iloc[0]),
        "trainable_correction_parameters": int(replication_frame.trainable_parameters.iloc[0]),
        "decision": {
            "regional": "reject: negligible seed-43 gain",
            "event_head": "defer: global-MAE gain remains modest and no evidence yet that event auxiliary loss improves the primary objective",
            "test_access": False,
        },
    }
    (ABLATION / "research_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

