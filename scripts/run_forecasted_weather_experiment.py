"""Compare persistence forcing with a learned causal weather-to-PM cascade."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def train(root: Path, args, mode: str) -> dict:
    output = root / args.output_dir / mode
    summary = output / "summary.json"
    if args.force or not summary.exists():
        command = [
            sys.executable, "-m", "common_local.train_dynamics",
            "--root", str(root),
            "--future-weather-mode", mode,
            "--disable-auxiliary", "--disable-month",
            "--seeds", *[str(seed) for seed in args.seeds],
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--weather-loss-weight", str(args.weather_loss_weight),
            "--output-dir", str(output.relative_to(root)),
            "--device", args.device,
        ]
        subprocess.run(command, cwd=root, check=True)
    return json.loads(summary.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/forecasted_weather")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weather-loss-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    persistence = train(root, args, "persistence")
    learned = train(root, args, "learned")
    persistence_mae = persistence["validation_mean_mae"]
    learned_mae = learned["validation_mean_mae"]
    result = {
        "experiment": "causal_forecasted_weather_to_pm_cascade",
        "selection_split": "KnowAir validation",
        "future_observed_weather_read": False,
        "seeds": args.seeds,
        "persistence_validation_mae": persistence_mae,
        "learned_weather_validation_mae": learned_mae,
        "learned_gain_over_persistence_mae": persistence_mae - learned_mae,
        "interpretation": (
            "Positive gain means forecasting weather improves PM2.5 MAE over "
            "repeating the last observed weather state."
        ),
        "test_accessed": False,
    }
    output = root / args.output_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
