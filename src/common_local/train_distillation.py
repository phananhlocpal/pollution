"""Train history-only TSR with privileged future-weather latent distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train_dynamics import run_seed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--panel-npz", help="External standardized panel NPZ, relative to root"
    )
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument(
        "--output-dir", default="artifacts/latent_impact_distillation"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--station-dim", type=int, default=8)
    parser.add_argument("--operator-dim", type=int, default=32)
    parser.add_argument("--max-step", type=float, default=.5)
    parser.add_argument("--distilled-latent-dim", type=int, default=16)
    parser.add_argument("--distilled-hidden-dim", type=int, default=32)
    parser.add_argument("--latent-kl-weight", type=float, default=.01)
    parser.add_argument(
        "--latent-samples", type=int, default=9,
        help="Deterministic prior scenarios aggregated by trajectory median",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--initialize-from")
    parser.add_argument("--disable-transport", action="store_true")
    parser.add_argument("--disable-source", action="store_true")
    parser.add_argument("--event-expert", action="store_true")
    return parser


def _complete_arguments(args):
    """Supply shared trainer options that are fixed by this protocol."""
    args.future_weather_mode = "distilled"
    args.gagnn_dir = None
    args.gagnn_protocol = "24x6"
    args.disable_auxiliary = True
    args.disable_month = True
    args.weather_hidden_dim = 16
    args.weather_loss_weight = 0.0
    args.weather_increment_loss_weight = 0.0
    args.transport_forcing_dim = 16
    args.source_forcing_dim = 32
    args.horizon_embedding_dim = 8
    args.seasonal_period = 8
    args.seasonal_cycles = 3
    return args


def main():
    args = _complete_arguments(build_parser().parse_args())
    rows = [run_seed(args, seed) for seed in args.seeds]
    validation_mae = [row["best_validation_mae"] for row in rows]
    summary = {
        "model": "latent_impact_distillation_tsr",
        "seeds": args.seeds,
        "validation_mae": validation_mae,
        "validation_mean_mae": float(np.mean(validation_mae)),
        "information_set_at_evaluation": "historical PM2.5 and meteorology only",
        "privileged_training_input": "realized future meteorology",
        "test_accessed": False,
    }
    output = Path(args.root) / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
