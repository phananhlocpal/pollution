"""Select learnable global source-regime memory on train-only rolling folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNITS = (8, 16, 32)
ADVANCEMENT_GAIN = 0.3
COMMON = (
    "--future-weather-mode", "factorized",
    "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
    "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
    "--horizon-embedding-dim", "8", "--weather-loss-weight", "0.1",
    "--weather-increment-loss-weight", "0.1", "--seeds", "43",
    "--epochs", "20", "--patience", "5", "--batch-size", "256", "--device", "cuda",
)


def run(*args):
    command = [sys.executable, "-m", "common_local.train_dynamics", *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summary(path: Path, command=None):
    if path.exists():
        print(f"REUSE {path.relative_to(ROOT)}", flush=True)
    elif command is not None:
        run(*command)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = []
    for fold in (1, 2, 3):
        baseline_path = ROOT / f"artifacts/rolling_delay/factorized_v3/fold_{fold}/summary.json"
        baseline = summary(baseline_path)["validation_mean_mae"]
        rows.append({"fold": fold, "units": 0, "validation_mae": baseline})
        for units in UNITS:
            output = f"artifacts/rolling_source_memory/units_{units}/fold_{fold}"
            path = ROOT / output / "summary.json"
            result = summary(path, (
                *COMMON, "--rolling-fold", str(fold),
                "--global-source-memory-units", str(units), "--output-dir", output,
            ))
            rows.append({
                "fold": fold, "units": units,
                "validation_mae": result["validation_mean_mae"],
            })
    baseline_mean = float(np.mean([
        row["validation_mae"] for row in rows if row["units"] == 0
    ]))
    aggregate = []
    for units in UNITS:
        selected = [row for row in rows if row["units"] == units]
        baselines = {
            row["fold"]: row["validation_mae"] for row in rows if row["units"] == 0
        }
        gains = [baselines[row["fold"]] - row["validation_mae"] for row in selected]
        aggregate.append({
            "units": units,
            "mean_mae": float(np.mean([row["validation_mae"] for row in selected])),
            "mean_gain": float(np.mean(gains)),
            "minimum_fold_gain": float(np.min(gains)),
            "all_folds_improve": bool(np.all(np.asarray(gains) > 0)),
        })
    selected = max(aggregate, key=lambda row: row["mean_gain"])
    advance = bool(
        selected["mean_gain"] >= ADVANCEMENT_GAIN and selected["all_folds_improve"]
    )
    payload = {
        "selection": "three rolling-origin folds inside KnowAir train only",
        "mechanism": "learnable global prototypes condition common source/sink only",
        "baseline_v3_mean_mae": baseline_mean,
        "fold_results": rows,
        "aggregate": aggregate,
        "selected": selected,
        "advancement_gate": "mean gain >=0.3 MAE and positive gain in every fold",
        "advance_to_original_validation": advance,
        "original_validation_accessed": False,
        "test_accessed": False,
    }
    decision = ROOT / "artifacts/rolling_source_memory/decision.json"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not advance:
        print("Global source memory failed rolling-fold gate; all tests remain sealed")
        return

    units = selected["units"]
    output = "artifacts/knowair_global_source_memory_selected"
    path = ROOT / output / "summary.json"
    result = summary(path, (
        "--future-weather-mode", "factorized", "--global-source-memory-units", str(units),
        "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
        "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
        "--horizon-embedding-dim", "8", "--weather-loss-weight", "0.1",
        "--weather-increment-loss-weight", "0.1", "--seeds", "42", "43", "44",
        "--epochs", "30", "--patience", "6", "--batch-size", "256",
        "--output-dir", output, "--device", "cuda",
    ))
    if result["validation_mean_mae"] > 18.0:
        print("Three-seed validation gate failed; KnowAir/China tests remain sealed")
        return
    print("Three-seed validation gate passed; freeze checkpoints before test access.")


if __name__ == "__main__":
    main()
