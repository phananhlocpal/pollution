"""Run the publication-grade validation ablation matrix sequentially."""

from __future__ import annotations

import subprocess
import sys


COMMON = [
    "--seeds", "42", "43", "44", "--epochs", "30", "--patience", "6",
    "--batch-size", "256", "--disable-auxiliary", "--disable-month",
    "--device", "cuda",
]


def run(*args):
    command = [sys.executable, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    run("-m", "common_local.train_dynamics", *COMMON,
        "--enable-lagged-transport",
        "--output-dir", "artifacts/retrained_ablation_full")
    run("-m", "common_local.train_dynamics", *COMMON,
        "--enable-lagged-transport", "--disable-transport",
        "--output-dir", "artifacts/retrained_ablation_no_transport")
    run("-m", "common_local.train_dynamics", *COMMON,
        "--enable-lagged-transport", "--disable-source",
        "--output-dir", "artifacts/retrained_ablation_no_source")
    run("-m", "common_local.train_dynamics", *COMMON, "--disable-lagged-transport",
        "--output-dir", "artifacts/retrained_ablation_no_lag")
    run(
        "-m", "common_local.train", "--seeds", "42", "43", "44",
        "--epochs", "30", "--patience", "6", "--batch-size", "256",
        "--lr", "0.003", "--scheduler", "--hidden-dim", "84",
        "--horizon-dim", "20", "--station-dim", "10", "--dropout", "0.1",
        "--gru-layers", "1", "--loss", "l1",
        "--output-dir", "artifacts/common_local_matched", "--device", "cuda",
    )
    run("scripts/summarize_retrained_ablations.py")


if __name__ == "__main__":
    main()
