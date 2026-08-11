"""Train, freeze, then evaluate corrected history-only China-AQI 96h->24h."""

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
        "scripts/audit_gagnn_reconstruction.py",
    )
    run(
        "-m", "common_local.train_dynamics",
        "--gagnn-dir", "data/benchmarks/china_aqi_gagnn",
        "--gagnn-protocol", "96x24", "--future-weather-mode", "latent",
        "--disable-lagged-transport", "--disable-auxiliary", "--disable-month",
        "--seeds", "42", "43", "44", "--epochs", "30", "--patience", "6",
        "--batch-size", "64", "--output-dir", "artifacts/china_aqi_96x24_latent_v2",
        "--device", "cuda",
    )
    run("scripts/freeze_china_aqi_96x24.py")
    run(
        "scripts/evaluate_china_aqi_96x24.py", "--allow-test",
        "--batch-size", "64", "--device", "cuda",
    )


if __name__ == "__main__":
    main()
