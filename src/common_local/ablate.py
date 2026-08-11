"""Train minimal residual corrections on frozen common_local checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from .correction import FrozenResidualCorrection
from .data import CommonLocalWindowDataset, load_panel
from .train import _loader, _run_epoch, choose_device


VARIANTS = {
    "spatial": ("spatial",),
    "wind": ("wind",),
    "meteo": ("meteo",),
    "regional": ("regional",),
    "wind_meteo": ("wind", "meteo"),
    "wind_spatial": ("wind", "spatial"),
}


def run_variant(args, variant, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(args.root); panel = load_panel(root); device = choose_device(args.device)
    model = FrozenResidualCorrection(
        VARIANTS[variant], root / "data/benchmarks/knowair/city.txt", panel.mean, panel.std
    ).to(device)
    base_path = root / f"artifacts/common_local/seed_{seed}/best_model.pt"
    model.base.load_state_dict(torch.load(base_path, map_location=device, weights_only=False)["model_state"])
    train_set = CommonLocalWindowDataset(panel, "train", args.max_train_samples)
    val_set = CommonLocalWindowDataset(panel, "val", args.max_eval_samples)
    train_loader = _loader(train_set, args.batch_size, seed, True)
    val_loader = _loader(val_set, args.batch_size, seed, False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    output = root / args.output_dir / variant / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "best_model.pt"
    best, stale, best_epoch, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, panel, device, optimizer)
        validation = _run_epoch(model, val_loader, panel, device)
        mae = validation["metrics"]["overall_1_72h"]["mae"]
        history.append({"epoch": epoch, "train_loss": train["loss"], "validation_mae": mae})
        print(json.dumps({"variant": variant, "seed": seed, **history[-1]}))
        if mae < best:
            best, stale, best_epoch = mae, 0, epoch
            torch.save({"model_state": model.state_dict(), "components": VARIANTS[variant]}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    validation = _run_epoch(model, val_loader, panel, device)
    baseline = json.loads((root / f"artifacts/common_local/seed_{seed}/metrics.json").read_text())
    baseline_mae = baseline["validation"]["metrics"]["overall_1_72h"]["mae"]
    payload = {
        "variant": variant, "components": VARIANTS[variant], "seed": seed,
        "best_epoch": best_epoch, "baseline_validation_mae": baseline_mae,
        "validation": validation, "delta_mae": best - baseline_mae,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()), "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/correction_ablation")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=(43,))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()
    rows = [run_variant(args, variant, seed) for variant in args.variants for seed in args.seeds]
    summary = [{"variant": row["variant"], "seed": row["seed"],
                "mae": row["validation"]["metrics"]["overall_1_72h"]["mae"],
                "delta_mae": row["delta_mae"]} for row in rows]
    output = Path(args.root) / args.output_dir; output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

