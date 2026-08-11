"""Wait for KnowAir ablations, then train/freeze/evaluate exact China-AQI."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    command = [sys.executable, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    ablation = ROOT / "artifacts/retrained_ablation/summary.json"
    while not ablation.exists():
        print("Waiting for retrained KnowAir ablations to release the GPU...", flush=True)
        time.sleep(10)
    run(
        "-m", "common_local.train_dynamics",
        "--gagnn-dir", "data/benchmarks/china_aqi_gagnn",
        "--future-weather-mode", "learned", "--disable-auxiliary", "--disable-month",
        "--seeds", "42", "43", "44", "--epochs", "30", "--patience", "6",
        "--batch-size", "256", "--output-dir", "artifacts/china_aqi_history_learned",
        "--device", "cuda",
    )
    run("scripts/freeze_china_aqi.py")
    run("scripts/evaluate_china_aqi.py", "--allow-test", "--batch-size", "256", "--device", "cuda")


if __name__ == "__main__":
    main()
