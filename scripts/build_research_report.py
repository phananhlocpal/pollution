"""Build final validation/test tables from frozen experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/predictions"


def evaluation(directory):
    return json.loads((PREDICTIONS / directory / "evaluation.json").read_text())


def metric_row(name, seed, split, report):
    metrics = report["metrics"]
    return {
        "model": name, "seed": seed, "split": split,
        "day1_mae": metrics["day1_1_24h"]["mae"],
        "day2_mae": metrics["day2_25_48h"]["mae"],
        "day3_mae": metrics["day3_49_72h"]["mae"],
        "mae": metrics["overall_1_72h"]["mae"],
        "rmse": metrics["overall_1_72h"]["rmse"],
        "smape": metrics["overall_1_72h"]["smape"],
        "persistence_skill_mae": metrics["skill_mae"]["overall_1_72h"],
    }


def main():
    rows = []
    for split in ("val", "test"):
        for seed in (42, 43, 44):
            base = evaluation(f"common_local_seed{seed}_{split}")
            rows.append(metric_row("common_local", seed, split, base))
            selected = evaluation(f"common_local_wind_meteo_seed{seed}_{split}")
            rows.append(metric_row("common_local+wind+meteo", seed, split, selected))
        air = evaluation(f"airdde_seed2024_{split}")
        rows.append(metric_row("AirDDE-repro", 2024, split, air))
    frame = pd.DataFrame(rows)
    output = ROOT / "artifacts/final_analysis"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    aggregate_rows = []
    for (model, split), group in frame.groupby(["model", "split"], sort=False):
        aggregate_rows.append({
            "model": model, "split": split, "seeds": len(group),
            **{f"{column}_mean": float(group[column].mean())
               for column in ("day1_mae", "day2_mae", "day3_mae", "mae", "rmse", "smape", "persistence_skill_mae")},
            "mae_std": float(group.mae.std(ddof=0)),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    val_base = json.loads((PREDICTIONS / "wind_meteo_vs_base_val.json").read_text())["comparison"]
    test_base = json.loads((PREDICTIONS / "wind_meteo_vs_base_test.json").read_text())["comparison"]
    val_air = json.loads((PREDICTIONS / "wind_meteo_vs_airdde_val.json").read_text())["comparison"]
    test_air = json.loads((PREDICTIONS / "wind_meteo_vs_airdde_test.json").read_text())["comparison"]
    selected_test = aggregate[(aggregate.model == "common_local+wind+meteo") & (aggregate.split == "test")].iloc[0]
    air_test = aggregate[(aggregate.model == "AirDDE-repro") & (aggregate.split == "test")].iloc[0]
    summary = {
        "frozen_model": "common_local+wind+meteo",
        "test_three_seed_mae_mean": float(selected_test.mae_mean),
        "test_three_seed_mae_std": float(selected_test.mae_std),
        "airdde_repro_test_mae_one_seed": float(air_test.mae_mean),
        "airdde_paper_reference_mae": 16.92,
        "paired_comparisons_seed43_vs_seed2024": {
            "selected_vs_base_validation": val_base,
            "selected_vs_base_test": test_base,
            "selected_vs_airdde_validation": val_air,
            "selected_vs_airdde_test": test_air,
        },
        "interpretation": [
            "The 257-parameter correction improves all three validation seeds and retains a smaller gain on test.",
            "AirDDE-repro remains materially more accurate; its local result is one seed and must not be conflated with the paper number.",
            "Regional correction was rejected and the event head was deferred based on validation evidence before test access.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    columns = ("model", "split", "seeds", "day1_mae_mean", "day2_mae_mean",
               "day3_mae_mean", "mae_mean", "mae_std", "rmse_mean", "smape_mean")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    table_rows = []
    for _, row in aggregate.iterrows():
        values = [str(row[column]) if column in ("model", "split") else
                  str(int(row[column])) if column == "seeds" else f"{row[column]:.4f}"
                  for column in columns]
        table_rows.append("| " + " | ".join(values) + " |")
    table = "\n".join((header, separator, *table_rows))
    markdown = f"""# Final residual-driven analysis\n\nArchitecture freeze: `frozen/wind_meteo/MANIFEST.json`.\n\n## Unified metrics\n\n{table}\n\n## Paired inference\n\n- Selected vs baseline validation: ΔMAE {val_base['mean_delta_mae_a_minus_b']:.4f}, CI95% {val_base['ci95']}.\n- Selected vs baseline test: ΔMAE {test_base['mean_delta_mae_a_minus_b']:.4f}, CI95% {test_base['ci95']}.\n- Selected vs AirDDE test: ΔMAE {test_air['mean_delta_mae_a_minus_b']:.4f}, CI95% {test_air['ci95']}.\n\nThe AirDDE comparison is one released-code seed (2024). Paper MAE 16.92 is retained only as an external reference.\n"""
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
