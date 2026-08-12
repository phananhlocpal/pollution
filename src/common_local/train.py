"""Train or evaluate the canonical common/local model on KnowAir validation."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CommonLocalWindowDataset, load_panel, load_standard_panel
from .losses import common_local_loss
from .metrics import validation_report
from .model import CommonLocalForecaster


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _loader(dataset, batch_size, seed, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _run_epoch(model, loader, panel, device, optimizer=None, loss_kind="l1"):
    training = optimizer is not None; model.train(training)
    loss_sum = batches = 0; predictions, truths, validity = [], [], []
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_batch(batch, device); output = model(batch)
            loss, _ = common_local_loss(
                output, batch["y"], float(panel.mean[0]), float(panel.std[0]), loss_kind,
                batch.get("y_valid"),
            )
            if "weather_prediction" in output and "future_weather_target" in batch:
                weather_loss = torch.nn.functional.smooth_l1_loss(
                    output["weather_prediction"], batch["future_weather_target"], beta=.5
                )
                loss = loss + model.weather_loss_weight * weather_loss
                if getattr(model, "weather_increment_loss_weight", 0.0) > 0:
                    previous = batch["x"][:, -1, :, 1:]
                    predicted_increment = torch.cat((
                        output["weather_prediction"][:, :1] - previous[:, None],
                        output["weather_prediction"][:, 1:]
                        - output["weather_prediction"][:, :-1],
                    ), 1)
                    target_increment = torch.cat((
                        batch["future_weather_target"][:, :1] - previous[:, None],
                        batch["future_weather_target"][:, 1:]
                        - batch["future_weather_target"][:, :-1],
                    ), 1)
                    increment_loss = torch.nn.functional.smooth_l1_loss(
                        predicted_increment, target_increment, beta=.5
                    )
                    loss = loss + model.weather_increment_loss_weight * increment_loss
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                predictions.append(output["prediction"].detach().cpu().numpy())
                truths.append(batch["y"].detach().cpu().numpy())
                if "y_valid" in batch:
                    validity.append(batch["y_valid"].detach().cpu().numpy())
            loss_sum += float(loss.detach()); batches += 1
    result = {"loss": loss_sum / max(batches, 1),
              "seconds": time.perf_counter() - started}
    if not training:
        prediction = np.concatenate(predictions) * panel.std[0] + panel.mean[0]
        truth = np.concatenate(truths) * panel.std[0] + panel.mean[0]
        result["metrics"] = validation_report(
            prediction, truth, cadence_hours=getattr(panel, "cadence_hours", 3),
            valid_mask=np.concatenate(validity) if validity else None,
        )
    return result


def run_seed(args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    panel = (
        load_standard_panel(Path(args.root) / args.panel_npz, args.expected_stations)
        if args.panel_npz else load_panel(args.root)
    )
    device = choose_device(args.device)
    train_set = CommonLocalWindowDataset(
        panel, "train", args.max_train_samples, args.history, args.horizon
    )
    val_set = CommonLocalWindowDataset(
        panel, "val", args.max_eval_samples, args.history, args.horizon
    )
    train_loader = _loader(train_set, args.batch_size, seed, True)
    val_loader = _loader(val_set, args.batch_size, seed, False)
    config = {
        "hidden_dim": args.hidden_dim, "horizon_dim": args.horizon_dim,
        "station_dim": args.station_dim, "dropout": args.dropout,
        "gru_layers": args.gru_layers,
        "weather_dim": panel.values.shape[-1] - 1,
        "horizon": args.horizon,
    }
    model = CommonLocalForecaster(stations=len(panel.stations), **config).to(device)
    output = Path(args.output_dir) / f"seed_{seed}"; output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.pt"
    parameters = sum(value.numel() for value in model.parameters())
    if args.evaluate_only:
        model.load_state_dict(torch.load(
            checkpoint, map_location=device, weights_only=False
        )["model_state"])
        validation = _run_epoch(model, val_loader, panel, device)
        print(json.dumps({"seed": seed, "validation": validation["metrics"]}))
        return validation

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if getattr(args, "scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=.5, patience=max(1, args.patience // 2), min_lr=1e-5
        )
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    print(f"model=common_local seed={seed} device={device} parameters={parameters:,} "
          f"train={len(train_set)} val={len(val_set)}")
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, panel, device, optimizer, args.loss)
        validation = _run_epoch(model, val_loader, panel, device, loss_kind=args.loss)
        mae = validation["metrics"]["overall_1_72h"]["mae"]
        if scheduler is not None:
            scheduler.step(mae)
        row = {"epoch": epoch, "train_loss": train["loss"],
               "validation_loss": validation["loss"], "validation_mae": mae,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row))
        if mae < best:
            best, best_epoch, stale = mae, epoch, 0
            torch.save({"model_state": model.state_dict(), "architecture": "common_local",
                        "config": config, "loss": args.loss}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=False
    )["model_state"])
    validation = _run_epoch(model, val_loader, panel, device, loss_kind=args.loss)
    payload = {
        "model": "common_local", "seed": seed, "parameter_count": parameters,
        "config": config, "loss": args.loss,
        "best_epoch": best_epoch, "best_validation_mae": best,
        "validation": validation, "history": history,
        "training_split": "first 70%" if args.panel_npz else "first 50%",
        "validation_split": "next 10%" if args.panel_npz else "next 25%",
        "test_accessed": False,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/common_local")
    parser.add_argument("--panel-npz", help="External standardized panel NPZ, relative to root")
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--horizon-dim", type=int, default=16)
    parser.add_argument("--station-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--loss", choices=("l1", "huber"), default="l1")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    rows = [run_seed(args, seed) for seed in args.seeds]
    metrics = [
        row["metrics"] if args.evaluate_only else row["validation"]["metrics"]
        for row in rows
    ]
    periods = ("day1_1_24h", "day2_25_48h", "day3_49_72h", "overall_1_72h")
    summary = {
        "model": "common_local",
        "seeds": args.seeds,
        "validation_mean": {
            period: {
                name: float(np.mean([row[period][name] for row in metrics]))
                for name in ("mae", "rmse", "smape")
            } for period in periods
        },
        "test_accessed": False,
        "development_subset": args.max_eval_samples is not None,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(output / "summary.json"),
                      "validation_mean": summary["validation_mean"]}, indent=2))


if __name__ == "__main__":
    main()
