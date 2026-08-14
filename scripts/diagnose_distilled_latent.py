"""Compare posterior-oracle and causal-prior rollouts on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import choose_device, move_batch


def _mae(model, loader, panel, device, latent_mode):
    absolute_error = 0.0
    valid_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model({**batch, "distilled_latent_mode": latent_mode})
            error = (output["prediction"] - batch["y"]).abs()
            valid = batch["y"] * float(panel.std[0]) + float(panel.mean[0]) >= 1e-4
            if "y_valid" in batch:
                valid = valid & batch["y_valid"].bool()
            error = error[valid]
            absolute_error += float(error.double().sum().cpu())
            valid_count += error.numel()
    return absolute_error / valid_count * float(panel.std[0])


def diagnose_checkpoint(checkpoint_path, city_path, panel, loader, device, samples):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "latent_impact_distillation_tsr":
        raise ValueError(f"Not a distilled checkpoint: {checkpoint_path}")
    model = TransportSourceRecurrentForecaster(
        city_path,
        stations=len(panel.stations), **checkpoint["config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.latent_samples = samples
    return {
        "posterior_oracle_mean_mae": _mae(
            model, loader, panel, device, "posterior_mean"
        ),
        "prior_mean_mae": _mae(model, loader, panel, device, "prior_mean"),
        f"prior_median_{samples}_mae": _mae(
            model, loader, panel, device, "prior_median"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/latent_impact_distillation"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--output", default="artifacts/latent_impact_distillation/diagnostics.json"
    )
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2 for the median diagnostic")

    root = Path(args.root).resolve()
    panel = load_panel(root)
    dataset = CommonLocalWindowDataset(
        panel, "val", args.max_eval_samples, history=24, horizon=24
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    device = choose_device(args.device)
    rows = {}
    for seed in args.seeds:
        checkpoint = root / args.checkpoint_dir / f"seed_{seed}/best_model.pt"
        rows[str(seed)] = diagnose_checkpoint(
            checkpoint, root / "data/benchmarks/knowair/city.txt",
            panel, loader, device, args.samples
        )
    keys = next(iter(rows.values()))
    result = {
        "split": "validation",
        "test_accessed": False,
        "samples": args.samples,
        "seeds": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows.values()])) for key in keys
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
