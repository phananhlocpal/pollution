"""Run a compact, validation-only capacity/loss search for common_local."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from common_local.train import run_seed


TRIALS = (
    {"name": "h64_l1", "hidden_dim": 64, "horizon_dim": 16, "station_dim": 8,
     "gru_layers": 1, "dropout": .10, "loss": "l1", "lr": 3e-3},
    {"name": "h96_l1", "hidden_dim": 96, "horizon_dim": 24, "station_dim": 12,
     "gru_layers": 1, "dropout": .15, "loss": "l1", "lr": 2e-3},
    {"name": "h64_l2", "hidden_dim": 64, "horizon_dim": 16, "station_dim": 8,
     "gru_layers": 2, "dropout": .10, "loss": "l1", "lr": 2e-3},
    {"name": "h64_huber", "hidden_dim": 64, "horizon_dim": 16, "station_dim": 8,
     "gru_layers": 1, "dropout": .10, "loss": "huber", "lr": 3e-3},
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/common_local_hpo")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--trials", nargs="*", choices=[row["name"] for row in TRIALS])
    cli = parser.parse_args()
    selected = [row for row in TRIALS if not cli.trials or row["name"] in cli.trials]
    results = []
    for trial in selected:
        arguments = Namespace(
            root=cli.root, output_dir=str(Path(cli.output_dir) / trial["name"]),
            max_train_samples=cli.max_train_samples, max_eval_samples=cli.max_eval_samples,
            batch_size=cli.batch_size, device=cli.device, epochs=cli.epochs,
            patience=cli.patience, weight_decay=cli.weight_decay,
            evaluate_only=False, scheduler=True,
            **{key: value for key, value in trial.items() if key != "name"},
        )
        payload = run_seed(arguments, cli.seed)
        results.append({
            "trial": trial["name"], "config": payload["config"],
            "loss": payload["loss"], "parameter_count": payload["parameter_count"],
            "best_epoch": payload["best_epoch"],
            "validation_mae": payload["best_validation_mae"],
            "test_accessed": False,
        })
    results.sort(key=lambda row: row["validation_mae"])
    output = Path(cli.root) / cli.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "search_summary.json").write_text(json.dumps({
        "selection_metric": "validation MAE", "seed": cli.seed,
        "test_accessed": False, "trials": results,
    }, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
