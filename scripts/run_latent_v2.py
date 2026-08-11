"""Validate history-only latent V2 on KnowAir, then run corrected China-AQI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWAIR_OUTPUT = ROOT / "artifacts/knowair_latent_v2"


def run(*args):
    command = [sys.executable, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    run(
        "-m", "common_local.train_dynamics",
        "--future-weather-mode", "latent", "--disable-lagged-transport",
        "--disable-auxiliary", "--disable-month",
        "--seeds", "42", "43", "44", "--epochs", "30", "--patience", "6",
        "--batch-size", "256", "--output-dir", "artifacts/knowair_latent_v2",
        "--device", "cuda",
    )
    summary = json.loads((KNOWAIR_OUTPUT / "summary.json").read_text())
    validation_mae = summary["validation_mean_mae"]
    if validation_mae > 18.0:
        raise SystemExit(
            f"KnowAir history-only V2 validation MAE {validation_mae:.4f} > 18.0; "
            "China-AQI test remains unopened."
        )
    run("scripts/run_china_aqi.py")


if __name__ == "__main__":
    main()
