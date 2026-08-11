"""Select adaptive delayed-state retrieval on train-only rolling folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON = (
    "--future-weather-mode", "factorized",
    "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
    "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
    "--horizon-embedding-dim", "8", "--weather-loss-weight", "0.1",
    "--weather-increment-loss-weight", "0.1", "--seeds", "43",
    "--epochs", "20", "--patience", "5", "--batch-size", "256",
    "--device", "cuda",
)
ADVANCEMENT_GAIN = 0.5


def run(*args):
    command = [sys.executable, "-m", "common_local.train_dynamics", *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = []
    for fold in (1, 2, 3):
        variants = {
            "factorized_v3": (),
            "adaptive_delay": ("--adaptive-delay", "--delay-dim", "16"),
        }
        for name, extra in variants.items():
            output = f"artifacts/rolling_delay/{name}/fold_{fold}"
            summary_path = ROOT / output / "summary.json"
            if summary_path.exists():
                print(f"REUSE {summary_path.relative_to(ROOT)}", flush=True)
            else:
                run(*COMMON, "--rolling-fold", str(fold), *extra, "--output-dir", output)
            summary = json.loads(summary_path.read_text())
            rows.append({
                "fold": fold, "variant": name,
                "validation_mae": summary["validation_mean_mae"],
                "test_accessed": summary["test_accessed"],
            })
    baseline = float(np.mean([
        row["validation_mae"] for row in rows if row["variant"] == "factorized_v3"
    ]))
    adaptive = float(np.mean([
        row["validation_mae"] for row in rows if row["variant"] == "adaptive_delay"
    ]))
    adaptive_gain = float(baseline - adaptive)
    payload = {
        "selection": "three rolling-origin folds inside KnowAir train only",
        "fold_results": rows,
        "mean_mae": {"factorized_v3": baseline, "adaptive_delay": adaptive},
        "adaptive_gain": adaptive_gain,
        "development_advancement_gain": ADVANCEMENT_GAIN,
        "advance_to_original_validation": bool(adaptive_gain >= ADVANCEMENT_GAIN),
        "test_accessed": False,
    }
    decision = ROOT / "artifacts/rolling_delay/decision.json"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not payload["advance_to_original_validation"]:
        print("Adaptive delay failed rolling-fold gate; all tests remain sealed")
        return

    run(
        "--future-weather-mode", "factorized", "--adaptive-delay", "--delay-dim", "16",
        "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
        "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
        "--horizon-embedding-dim", "8", "--weather-loss-weight", "0.1",
        "--weather-increment-loss-weight", "0.1", "--seeds", "42", "43", "44",
        "--epochs", "30", "--patience", "6", "--batch-size", "256",
        "--output-dir", "artifacts/knowair_adaptive_delay_selected", "--device", "cuda",
    )
    summary = json.loads(
        (ROOT / "artifacts/knowair_adaptive_delay_selected/summary.json").read_text()
    )
    if summary["validation_mean_mae"] > 18.0:
        raise SystemExit("Three-seed validation gate failed; KnowAir/China tests remain sealed")
    print("Three-seed validation gate passed; freeze before any test access.")


if __name__ == "__main__":
    main()
