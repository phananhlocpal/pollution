"""Frozen-checkpoint knockout ablations for the recurrent operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import _loader, _run_epoch, choose_device


VARIANTS = {
    "full": {},
    "no_transport": {"use_transport": False},
    "no_source": {"use_source": False},
    "no_lagged_transport": {"use_lagged_transport": False},
    "no_month": {"use_month": False},
    "strict_features": {"use_auxiliary": False, "use_month": False},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="artifacts/transport_source_recurrent/knockout_ablation.json")
    args = parser.parse_args()
    root = Path(args.root); panel = load_panel(root); device = choose_device(args.device)
    dataset = CommonLocalWindowDataset(panel, "val")
    rows = []
    for seed in args.seeds:
        loader = _loader(dataset, args.batch_size, seed, False)
        checkpoint = torch.load(
            root / f"artifacts/transport_source_recurrent/seed_{seed}/best_model.pt",
            map_location=device, weights_only=False,
        )
        for variant, overrides in VARIANTS.items():
            config = {**checkpoint.get("config", {}), **overrides}
            model = TransportSourceRecurrentForecaster(
                root / "data/benchmarks/knowair/city.txt",
                stations=len(panel.stations), **config,
            ).to(device)
            incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
            missing = [key for key in incompatible.missing_keys if key != "station_threshold"]
            if missing or incompatible.unexpected_keys:
                raise ValueError(f"Incompatible checkpoint: {incompatible}")
            metrics = _run_epoch(model, loader, panel, device)["metrics"]
            rows.append({
                "seed": seed, "variant": variant,
                "mae": metrics["overall_1_72h"]["mae"],
                "day1_mae": metrics["day1_1_24h"]["mae"],
                "day2_mae": metrics["day2_25_48h"]["mae"],
                "day3_mae": metrics["day3_49_72h"]["mae"],
            })
            del model; torch.cuda.empty_cache()
    full = {row["seed"]: row["mae"] for row in rows if row["variant"] == "full"}
    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append({
            "variant": variant,
            "mean_mae": float(np.mean([row["mae"] for row in selected])),
            "mean_delta_mae_vs_full": float(np.mean([
                row["mae"] - full[row["seed"]] for row in selected
            ])),
            "mean_day_mae": [float(np.mean([row[key] for row in selected])) for key in (
                "day1_mae", "day2_mae", "day3_mae"
            )],
        })
    result = {
        "type": "frozen_checkpoint_knockout",
        "selection_split": "validation", "test_accessed": False,
        "rows": rows, "summary": summary,
    }
    output = root / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
