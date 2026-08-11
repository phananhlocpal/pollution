"""Screen causal daily-cycle weather forcing on KnowAir validation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = (
    "--disable-lagged-transport", "--disable-auxiliary", "--disable-month", "--seeds", "43",
    "--epochs", "30", "--patience", "6", "--batch-size", "256",
    "--seasonal-period", "8", "--device", "cuda",
)


def run(*args):
    command = [sys.executable, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    variants = {
        "seasonal": "artifacts/knowair_weather_seasonal",
        "seasonal_weighted": "artifacts/knowair_weather_seasonal_weighted",
    }
    rows = {}
    for mode, output in variants.items():
        run(
            "-m", "common_local.train_dynamics", "--future-weather-mode", mode,
            *COMMON, "--output-dir", output,
        )
        summary = json.loads((ROOT / output / "summary.json").read_text())
        rows[mode] = summary["validation_mean_mae"]
    selected = min(rows, key=rows.get)
    payload = {
        "selection_split": "KnowAir validation",
        "seed43_screen_mae": rows,
        "selected": selected,
        "screen_gate_max_mae": 18.0,
        "screen_passed": rows[selected] <= 18.0,
        "test_accessed": False,
    }
    output = ROOT / "artifacts/seasonal_forcing/screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not payload["screen_passed"]:
        raise SystemExit("Seasonal forcing failed validation screen; all tests remain sealed")

    selected_output = "artifacts/knowair_seasonal_selected"
    run(
        "-m", "common_local.train_dynamics", "--future-weather-mode", selected,
        "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
        "--seeds", "42", "43", "44",
        "--epochs", "30", "--patience", "6", "--batch-size", "256",
        "--seasonal-period", "8", "--output-dir", selected_output, "--device", "cuda",
    )
    summary = json.loads((ROOT / selected_output / "summary.json").read_text())
    if summary["validation_mean_mae"] > 18.0:
        raise SystemExit("Three-seed seasonal forcing failed gate; all tests remain sealed")
    print("Three-seed KnowAir gate passed; freeze before any test evaluation.")


if __name__ == "__main__":
    main()
