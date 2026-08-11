"""Screen factorized exogenous V3 on sealed KnowAir validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = (
    "--future-weather-mode", "factorized",
    "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
    "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
    "--horizon-embedding-dim", "8", "--seeds", "43",
    "--epochs", "30", "--patience", "6", "--batch-size", "256",
    "--device", "cuda",
)


def run(*args):
    command = [sys.executable, "-m", "common_local.train_dynamics", *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    variants = {
        "pm_only": (0.0, 0.0, "artifacts/knowair_factorized_v3_pm_only"),
        "system_identification": (0.1, 0.1, "artifacts/knowair_factorized_v3_system_id"),
    }
    rows = {}
    for name, (weather_weight, increment_weight, output) in variants.items():
        run(
            *COMMON,
            "--weather-loss-weight", str(weather_weight),
            "--weather-increment-loss-weight", str(increment_weight),
            "--output-dir", output,
        )
        summary = json.loads((ROOT / output / "summary.json").read_text())
        rows[name] = summary["validation_mean_mae"]

    selected = min(rows, key=rows.get)
    payload = {
        "selection_split": "KnowAir validation",
        "seed43_screen_mae": rows,
        "selected": selected,
        "screen_gate_max_mae": 18.0,
        "screen_passed": rows[selected] <= 18.0,
        "explicit_lag": False,
        "test_accessed": False,
    }
    screen = ROOT / "artifacts/factorized_v3/screen.json"
    screen.parent.mkdir(parents=True, exist_ok=True)
    screen.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not payload["screen_passed"]:
        raise SystemExit("V3 failed validation screen; KnowAir and China tests remain sealed")

    weather_weight, increment_weight, _ = variants[selected]
    run(
        "--future-weather-mode", "factorized",
        "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
        "--transport-forcing-dim", "16", "--source-forcing-dim", "32",
        "--horizon-embedding-dim", "8", "--seeds", "42", "43", "44",
        "--epochs", "30", "--patience", "6", "--batch-size", "256",
        "--weather-loss-weight", str(weather_weight),
        "--weather-increment-loss-weight", str(increment_weight),
        "--output-dir", "artifacts/knowair_factorized_v3_selected", "--device", "cuda",
    )
    summary = json.loads(
        (ROOT / "artifacts/knowair_factorized_v3_selected/summary.json").read_text()
    )
    if summary["validation_mean_mae"] > 18.0:
        raise SystemExit("V3 three-seed gate failed; KnowAir and China tests remain sealed")
    print("V3 three-seed KnowAir gate passed; freeze checkpoints before test access.")


if __name__ == "__main__":
    main()
