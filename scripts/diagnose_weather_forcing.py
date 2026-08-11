"""Measure causal weather skill and oracle-channel PM sensitivity on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import _loader, choose_device, move_batch


FEATURES = (
    "temperature", "pressure", "relative_humidity", "wind_speed",
    "wind_direction_sin", "wind_direction_cos",
)
GROUPS = {
    "temperature": (0,),
    "pressure": (1,),
    "relative_humidity": (2,),
    "wind_speed": (3,),
    "wind_direction_vector": (4, 5),
    "all_weather": (0, 1, 2, 3, 4, 5),
}


def _mae_record(total, count, cadence=3):
    horizon = total.shape[0]
    result = {"overall": float(total.sum() / count.sum())}
    per_day = 24 // cadence
    for day, left in enumerate(range(0, horizon, per_day), 1):
        right = min(horizon, left + per_day)
        result[f"day_{day}"] = float(total[left:right].sum() / count[left:right].sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/transport_source_recurrent_history_learned/seed_43/best_model.pt",
    )
    parser.add_argument(
        "--output", default="artifacts/weather_diagnostics/learned_seed43_validation.json"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root)
    panel = load_panel(root)
    dataset = CommonLocalWindowDataset(panel, "val")
    device = choose_device(args.device)
    checkpoint = torch.load(root / args.checkpoint, map_location=device, weights_only=False)
    forcing_mode = checkpoint["config"].get("future_weather_mode")
    if forcing_mode not in {"learned", "factorized"}:
        raise SystemExit("Weather diagnostics require a causal weather-decoding checkpoint")
    supports_oracle_substitution = forcing_mode == "learned"
    model = TransportSourceRecurrentForecaster(
        root / "data/benchmarks/knowair/city.txt",
        stations=len(panel.stations), coordinates=panel.coordinates,
        **checkpoint["config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    horizon, weather_dim = model.horizon, len(FEATURES)
    sums = {
        name: np.zeros((horizon, weather_dim), dtype=np.float64)
        for name in ("error2", "pred", "truth", "pred2", "truth2", "product", "count")
    }
    angle_error = np.zeros(horizon, dtype=np.float64)
    angle_count = np.zeros(horizon, dtype=np.float64)
    variants = {"causal": np.zeros(horizon, dtype=np.float64)}
    variant_counts = {"causal": np.zeros(horizon, dtype=np.float64)}
    if supports_oracle_substitution:
        for name in GROUPS:
            variants[f"oracle_{name}"] = np.zeros(horizon, dtype=np.float64)
            variant_counts[f"oracle_{name}"] = np.zeros(horizon, dtype=np.float64)

    weather_mean = panel.mean[1:].astype(np.float32)
    weather_std = panel.std[1:].astype(np.float32)
    target_mean = float(panel.mean[0])
    target_std = float(panel.std[0])
    with torch.no_grad():
        for batch in _loader(dataset, args.batch_size, 43, False):
            batch = move_batch(batch, device)
            baseline = model(batch)
            predicted_norm = baseline["weather_prediction"]
            truth_norm = batch["future_weather_target"]
            predicted = predicted_norm.cpu().numpy() * weather_std + weather_mean
            truth = truth_norm.cpu().numpy() * weather_std + weather_mean
            axes = (0, 2)
            sums["error2"] += np.square(predicted - truth).sum(axis=axes)
            sums["pred"] += predicted.sum(axis=axes)
            sums["truth"] += truth.sum(axis=axes)
            sums["pred2"] += np.square(predicted).sum(axis=axes)
            sums["truth2"] += np.square(truth).sum(axis=axes)
            sums["product"] += (predicted * truth).sum(axis=axes)
            sums["count"] += predicted.shape[0] * predicted.shape[2]

            predicted_angle = np.arctan2(predicted[..., 4], predicted[..., 5])
            truth_angle = np.arctan2(truth[..., 4], truth[..., 5])
            difference = np.angle(np.exp(1j * (predicted_angle - truth_angle)))
            angle_error += np.abs(np.degrees(difference)).sum(axis=(0, 2))
            angle_count += predicted.shape[0] * predicted.shape[2]

            target = batch["y"]
            valid = target * target_std + target_mean >= 1e-4
            error = torch.abs(baseline["prediction"] - target) * target_std
            variants["causal"] += (error * valid).sum(dim=(0, 2)).cpu().numpy()
            variant_counts["causal"] += valid.sum(dim=(0, 2)).cpu().numpy()
            if supports_oracle_substitution:
                for name, indices in GROUPS.items():
                    override = predicted_norm.clone()
                    override[..., list(indices)] = truth_norm[..., list(indices)]
                    diagnostic_batch = dict(batch)
                    diagnostic_batch["diagnostic_future_weather_override"] = override
                    output = model(diagnostic_batch)
                    error = torch.abs(output["prediction"] - target) * target_std
                    key = f"oracle_{name}"
                    variants[key] += (error * valid).sum(dim=(0, 2)).cpu().numpy()
                    variant_counts[key] += valid.sum(dim=(0, 2)).cpu().numpy()

    rmse = np.sqrt(sums["error2"] / sums["count"])
    numerator = sums["product"] - sums["pred"] * sums["truth"] / sums["count"]
    pred_var = sums["pred2"] - np.square(sums["pred"]) / sums["count"]
    truth_var = sums["truth2"] - np.square(sums["truth"]) / sums["count"]
    correlation = numerator / np.sqrt(np.maximum(pred_var * truth_var, 1e-12))
    selected_hours = (3, 6, 12, 24, 48, 72)
    indices = [hour // 3 - 1 for hour in selected_hours]
    weather_skill = {}
    for feature, column in zip(FEATURES, range(weather_dim)):
        weather_skill[feature] = {
            str(hour): {"rmse": float(rmse[index, column]), "correlation": float(correlation[index, column])}
            for hour, index in zip(selected_hours, indices)
        }
    weather_skill["wind_direction"] = {
        str(hour): {"mean_absolute_angular_error_degrees": float(angle_error[index] / angle_count[index])}
        for hour, index in zip(selected_hours, indices)
    }
    pm_mae = {
        name: _mae_record(values, variant_counts[name]) for name, values in variants.items()
    }
    baseline_mae = pm_mae["causal"]["overall"]
    oracle_gain = {
        name.removeprefix("oracle_"): baseline_mae - row["overall"]
        for name, row in pm_mae.items() if name.startswith("oracle_")
    }
    payload = {
        "split": "KnowAir validation only",
        "seed": 43,
        "checkpoint": args.checkpoint,
        "future_weather_mode": forcing_mode,
        "future_realized_weather_used_by_causal_baseline": False,
        "oracle_substitution_role": (
            "diagnostic only; never eligible as a forecasting result"
            if supports_oracle_substitution else
            "not run: V3 source/sink consumes latent qS directly, not decoded weather"
        ),
        "weather_skill_by_forecast_hour": weather_skill,
        "pm_mae": pm_mae,
        "oracle_channel_mae_gain": oracle_gain,
        "test_accessed": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
