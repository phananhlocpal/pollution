"""Train the validation-only Transport--Source Recurrent Operator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .analog_memory import rolling_origin_folds
from .data import (
    CommonLocalOriginDataset, CommonLocalWindowDataset,
    GAGNNAirDDEWindowDataset, GAGNNWindowDataset,
    audit_gagnn_overlap, fit_seasonal_weather_weights, load_gagnn_metadata,
    load_panel, load_standard_panel,
)
from .dynamics import TransportSourceRecurrentForecaster
from .train import _loader, _run_epoch, choose_device


def run_seed(args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(args.root)
    if args.gagnn_dir:
        if args.future_weather_mode == "observed":
            raise ValueError("GAGNN provides historical covariates only; choose persistence, learned, or latent")
        if args.gagnn_protocol == "96x24" and args.future_weather_mode not in {
            "latent", "factorized"
        }:
            raise ValueError("Corrected GAGNN 96x24 runs require history-only forcing")
        if args.gagnn_protocol == "96x24":
            audit = audit_gagnn_overlap(root / args.gagnn_dir)
            if not audit["reconstructable"]:
                raise ValueError("GAGNN windows do not support exact split-local reconstruction")
        panel = load_gagnn_metadata(root / args.gagnn_dir, args.gagnn_protocol)
    else:
        panel = (
            load_standard_panel(root / args.panel_npz, args.expected_stations)
            if args.panel_npz else load_panel(root)
        )
    device = choose_device(args.device)
    if args.gagnn_dir:
        dataset_type = (
            GAGNNAirDDEWindowDataset if args.gagnn_protocol == "96x24"
            else GAGNNWindowDataset
        )
        train_set = dataset_type(root / args.gagnn_dir, "train", panel, args.max_train_samples)
        val_set = dataset_type(root / args.gagnn_dir, "val", panel, args.max_eval_samples)
        history, horizon = panel.history, panel.horizon
    else:
        if args.rolling_fold:
            fold = rolling_origin_folds(
                panel.split_points[0], args.history, args.horizon
            )[args.rolling_fold - 1]
            train_set = CommonLocalOriginDataset(
                panel, fold.candidate_origins, args.history, args.horizon,
                args.max_train_samples,
            )
            val_set = CommonLocalOriginDataset(
                panel, fold.query_origins, args.history, args.horizon,
                args.max_eval_samples,
            )
        else:
            train_set = CommonLocalWindowDataset(
                panel, "train", args.max_train_samples, args.history, args.horizon
            )
            val_set = CommonLocalWindowDataset(
                panel, "val", args.max_eval_samples, args.history, args.horizon
            )
        history, horizon = args.history, args.horizon
    train_loader = _loader(train_set, args.batch_size, seed, True)
    val_loader = _loader(val_set, args.batch_size, seed, False)
    config = {
        "hidden_dim": args.hidden_dim,
        "station_dim": args.station_dim,
        "operator_dim": args.operator_dim,
        "max_step": args.max_step,
        "event_expert": args.event_expert,
        "use_transport": not args.disable_transport,
        "use_source": not args.disable_source,
        "use_lagged_transport": not args.disable_lagged_transport,
        "use_auxiliary": not args.disable_auxiliary,
        "use_month": not args.disable_month,
        "future_weather_mode": args.future_weather_mode,
        "weather_hidden_dim": args.weather_hidden_dim,
        "weather_loss_weight": args.weather_loss_weight,
        "weather_increment_loss_weight": args.weather_increment_loss_weight,
        "transport_forcing_dim": args.transport_forcing_dim,
        "source_forcing_dim": args.source_forcing_dim,
        "horizon_embedding_dim": args.horizon_embedding_dim,
        "use_adaptive_delay": args.adaptive_delay,
        "delay_dim": args.delay_dim,
        "seasonal_period": args.seasonal_period,
        "horizon": horizon,
    }
    if args.future_weather_mode == "seasonal_weighted":
        if args.gagnn_dir:
            raise ValueError("Train-fitted weighted seasonal forcing is currently KnowAir-only")
        config["seasonal_weights"] = fit_seasonal_weather_weights(
            panel, args.seasonal_period, args.seasonal_cycles
        ).tolist()
    config["weather_dim"] = panel.weather_dim if args.gagnn_dir else panel.values.shape[-1] - 1
    model = TransportSourceRecurrentForecaster(
        root / "data/benchmarks/knowair/city.txt" if not (args.panel_npz or args.gagnn_dir) else None,
        stations=len(panel.stations), coordinates=panel.coordinates, **config,
    ).to(device)
    threshold = panel.target_threshold if args.gagnn_dir else np.quantile(
        panel.values[:panel.split_points[0], :, 0], .9, axis=0
    )
    model.station_threshold.copy_(torch.as_tensor(threshold, dtype=torch.float32, device=device))
    if args.initialize_from:
        initial = torch.load(root / args.initialize_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model_state"], strict=False)
        if unexpected:
            raise ValueError(f"Unexpected warm-start keys: {unexpected}")
        print(f"warm_start={args.initialize_from} missing={missing}")
    parameters = sum(value.numel() for value in model.parameters())
    output = root / args.output_dir / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.pt"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=max(1, args.patience // 2), min_lr=1e-5
    )
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    print(
        f"model=transport_source_recurrent seed={seed} device={device} "
        f"parameters={parameters:,} train={len(train_set)} val={len(val_set)}"
    )
    overall_key = f"overall_1_{horizon * getattr(panel, 'cadence_hours', 3)}h"
    architecture = {
        "latent": "latent_forcing_transport_source_recurrent_v2",
        "factorized": "factorized_exogenous_transport_source_v3",
    }.get(args.future_weather_mode, "transport_source_recurrent")
    if args.adaptive_delay:
        architecture = "adaptive_delayed_transport_source_v4"
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, panel, device, optimizer)
        validation = _run_epoch(model, val_loader, panel, device)
        mae = validation["metrics"][overall_key]["mae"]
        scheduler.step(mae)
        row = {
            "epoch": epoch, "train_loss": train["loss"],
            "validation_loss": validation["loss"], "validation_mae": mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if mae < best:
            best, best_epoch, stale = mae, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "architecture": architecture,
                "config": config,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    validation = _run_epoch(model, val_loader, panel, device)
    payload = {
        "model": architecture, "seed": seed,
        "config": config, "parameter_count": parameters,
        "best_epoch": best_epoch, "best_validation_mae": best,
        "validation": validation, "history": history,
        "training_split": (
            "official released train split" if args.gagnn_dir else
            f"rolling train fold {args.rolling_fold}" if args.rolling_fold else "first 50%"
        ),
        "validation_split": (
            "official released validation split" if args.gagnn_dir else
            f"rolling dev fold {args.rolling_fold}" if args.rolling_fold else "next 25%"
        ),
        "gagnn_protocol": args.gagnn_protocol if args.gagnn_dir else None,
        "test_accessed": False,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--panel-npz", help="External standardized lockbox NPZ, relative to root")
    parser.add_argument("--gagnn-dir", help="Official pre-windowed 209-city GAGNN data directory")
    parser.add_argument("--gagnn-protocol", choices=("24x6", "96x24"), default="24x6")
    parser.add_argument("--expected-stations", type=int, help="Fail if an external lockbox has a different node count")
    parser.add_argument("--output-dir", default="artifacts/transport_source_recurrent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--rolling-fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--station-dim", type=int, default=8)
    parser.add_argument("--operator-dim", type=int, default=32)
    parser.add_argument("--max-step", type=float, default=.5)
    parser.add_argument("--event-expert", action="store_true")
    parser.add_argument("--initialize-from")
    parser.add_argument("--disable-transport", action="store_true")
    parser.add_argument("--disable-source", action="store_true")
    parser.add_argument("--disable-lagged-transport", action="store_true")
    parser.add_argument("--disable-auxiliary", action="store_true")
    parser.add_argument("--disable-month", action="store_true")
    parser.add_argument(
        "--future-weather-mode",
        choices=(
            "observed", "persistence", "learned", "latent", "seasonal",
            "seasonal_weighted", "factorized",
        ),
                        default="observed")
    parser.add_argument("--weather-hidden-dim", type=int, default=16)
    parser.add_argument("--weather-loss-weight", type=float, default=.1)
    parser.add_argument("--weather-increment-loss-weight", type=float, default=0.0)
    parser.add_argument("--transport-forcing-dim", type=int, default=16)
    parser.add_argument("--source-forcing-dim", type=int, default=32)
    parser.add_argument("--horizon-embedding-dim", type=int, default=8)
    parser.add_argument("--adaptive-delay", action="store_true")
    parser.add_argument("--delay-dim", type=int, default=16)
    parser.add_argument("--seasonal-period", type=int, default=8)
    parser.add_argument("--seasonal-cycles", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()
    rows = [run_seed(args, seed) for seed in args.seeds]
    summary = {
        "model": rows[0]["model"], "seeds": args.seeds,
        "validation_mae": [row["best_validation_mae"] for row in rows],
        "validation_mean_mae": float(np.mean([row["best_validation_mae"] for row in rows])),
        "test_accessed": False,
    }
    output = Path(args.root) / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
