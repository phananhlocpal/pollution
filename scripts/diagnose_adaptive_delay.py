"""Summarize same-station adaptive-delay attention on rolling dev folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common_local.analog_memory import rolling_origin_folds
from common_local.data import CommonLocalOriginDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import _loader, choose_device, move_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/rolling_delay/attention_diagnostics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root)
    panel = load_panel(root)
    device = choose_device(args.device)
    lag_values = np.arange(24, 0, -1, dtype=np.float64)
    folds_payload = []
    for fold_index, fold in enumerate(rolling_origin_folds(panel.split_points[0]), 1):
        checkpoint_path = root / (
            f"artifacts/rolling_delay/adaptive_delay/fold_{fold_index}/"
            "seed_43/best_model.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = TransportSourceRecurrentForecaster(
            root / "data/benchmarks/knowair/city.txt", stations=len(panel.stations),
            coordinates=panel.coordinates, **checkpoint["config"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        dataset = CommonLocalOriginDataset(panel, fold.query_origins)
        entropy_sum = np.zeros(24); lag_sum = np.zeros(24); count = np.zeros(24)
        top_counts = np.zeros((3, 24), dtype=np.int64)
        regime = {
            name: {"entropy": 0.0, "effective_lag": 0.0, "count": 0}
            for name in ("pm_regular", "pm_high", "wind_calm", "wind_moderate", "wind_strong")
        }
        with torch.no_grad():
            for batch in _loader(dataset, args.batch_size, 43, False):
                history = batch["x"].numpy()
                batch = move_batch(batch, device)
                batch["diagnostic_delay_attention"] = True
                attention = model(batch)["delay_attention"].cpu().numpy()
                entropy = -(attention * np.log(np.maximum(attention, 1e-12))).sum(-1)
                entropy /= np.log(attention.shape[-1])
                effective = np.einsum("bhnl,l->bhn", attention, lag_values)
                entropy_sum += entropy.sum((0, 2)); lag_sum += effective.sum((0, 2))
                count += entropy.shape[0] * entropy.shape[2]
                top = lag_values[np.argmax(attention, axis=-1)].astype(int)
                for day in range(3):
                    values = top[:, day * 8:(day + 1) * 8].reshape(-1)
                    top_counts[day] += np.bincount(values - 1, minlength=24)

                last_pm = history[:, -1, :, 0]
                high = last_pm >= model.station_threshold.cpu().numpy()[None]
                wind = history[:, -1, :, 4] * panel.std[4] + panel.mean[4]
                masks = {
                    "pm_regular": ~high, "pm_high": high,
                    "wind_calm": wind < 1,
                    "wind_moderate": (wind >= 1) & (wind < 3),
                    "wind_strong": wind >= 3,
                }
                for name, mask in masks.items():
                    expanded = np.broadcast_to(mask[:, None], entropy.shape)
                    regime[name]["entropy"] += float(entropy[expanded].sum())
                    regime[name]["effective_lag"] += float(effective[expanded].sum())
                    regime[name]["count"] += int(expanded.sum())
        for row in regime.values():
            row["mean_entropy"] = row.pop("entropy") / max(row["count"], 1)
            row["mean_effective_lag_steps"] = row.pop("effective_lag") / max(row["count"], 1)
        folds_payload.append({
            "fold": fold_index,
            "normalized_entropy_by_horizon": (entropy_sum / count).tolist(),
            "effective_lag_steps_by_horizon": (lag_sum / count).tolist(),
            "top_lag_counts_by_day": top_counts.tolist(),
            "regimes": regime,
        })
    payload = {
        "split": "three KnowAir train-internal rolling dev folds",
        "role": "diagnostic only; no architecture reselection",
        "folds": folds_payload,
        "original_validation_accessed": False,
        "test_accessed": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
