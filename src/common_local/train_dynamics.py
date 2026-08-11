"""Train the validation-only Transport--Source Recurrent Operator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import CommonLocalWindowDataset, load_panel
from .dynamics import TransportSourceRecurrentForecaster
from .train import _loader, _run_epoch, choose_device


def run_seed(args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(args.root)
    panel = load_panel(root)
    device = choose_device(args.device)
    train_set = CommonLocalWindowDataset(panel, "train", args.max_train_samples)
    val_set = CommonLocalWindowDataset(panel, "val", args.max_eval_samples)
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
    }
    model = TransportSourceRecurrentForecaster(
        root / "data/benchmarks/knowair/city.txt",
        stations=len(panel.stations), **config,
    ).to(device)
    train_end = panel.split_points[0]
    model.station_threshold.copy_(torch.as_tensor(
        np.quantile(panel.values[:train_end, :, 0], .9, axis=0),
        dtype=torch.float32, device=device,
    ))
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
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, panel, device, optimizer)
        validation = _run_epoch(model, val_loader, panel, device)
        mae = validation["metrics"]["overall_1_72h"]["mae"]
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
                "architecture": "transport_source_recurrent",
                "config": config,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    validation = _run_epoch(model, val_loader, panel, device)
    payload = {
        "model": "transport_source_recurrent", "seed": seed,
        "config": config, "parameter_count": parameters,
        "best_epoch": best_epoch, "best_validation_mae": best,
        "validation": validation, "history": history,
        "training_split": "first 50%", "validation_split": "next 25%",
        "test_accessed": False,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/transport_source_recurrent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()
    rows = [run_seed(args, seed) for seed in args.seeds]
    summary = {
        "model": "transport_source_recurrent", "seeds": args.seeds,
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
