"""Probe train-only historical regime retrieval before building neural memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common_local.analog_memory import (
    analog_continuation_metrics, global_keys, inverse_distance_weights,
    retrieve_neighbors, rolling_origin_folds,
)
from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import _loader, choose_device, move_batch


KEYS = ("sequence", "multiscale")
NEIGHBORS = (1, 4, 8)
V3_REFERENCE_MAE = 20.569299697875977
ADVANCEMENT_GAIN = 0.5


def _subsample(origins: np.ndarray, maximum: int) -> np.ndarray:
    if len(origins) <= maximum:
        return origins
    return origins[np.linspace(0, len(origins) - 1, maximum, dtype=int)]


def rolling_screen(panel, maximum_queries: int) -> dict:
    rows = []
    for fold_index, fold in enumerate(
        rolling_origin_folds(panel.split_points[0]), 1
    ):
        queries = _subsample(fold.query_origins, maximum_queries)
        for key_name in KEYS:
            candidate_keys = global_keys(panel.values, fold.candidate_origins, 24, key_name)
            query_keys = global_keys(panel.values, queries, 24, key_name)
            indices, distances = retrieve_neighbors(candidate_keys, query_keys, max(NEIGHBORS))
            for neighbors in NEIGHBORS:
                metrics = analog_continuation_metrics(
                    panel.values, fold.candidate_origins, queries,
                    indices[:, :neighbors], distances[:, :neighbors],
                    float(panel.mean[0]), float(panel.std[0]),
                )
                rows.append({
                    "fold": fold_index,
                    "candidate_count": len(fold.candidate_origins),
                    "query_count": len(queries),
                    "key": key_name,
                    "neighbors": neighbors,
                    **metrics,
                })
    aggregate = []
    for key_name in KEYS:
        for neighbors in NEIGHBORS:
            selected = [
                row for row in rows
                if row["key"] == key_name and row["neighbors"] == neighbors
            ]
            aggregate.append({
                "key": key_name,
                "neighbors": neighbors,
                **{
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in (
                        "pm_increment_mae", "pm_persistence_mae",
                        "weather_normalized_rmse", "weather_persistence_normalized_rmse",
                    )
                },
            })
    weather = min(aggregate, key=lambda row: row["weather_normalized_rmse"])
    pm = min(aggregate, key=lambda row: row["pm_increment_mae"])
    return {"folds": rows, "aggregate": aggregate, "selected_weather": weather,
            "selected_pm_increment": pm}


def retrieval(panel, origins, key_name, neighbors):
    train_end = panel.split_points[0]
    candidates = np.arange(24, train_end - 24 + 1, dtype=np.int64)
    candidate_keys = global_keys(panel.values, candidates, 24, key_name)
    query_keys = global_keys(panel.values, origins, 24, key_name)
    indices, distances = retrieve_neighbors(candidate_keys, query_keys, neighbors)
    return candidates, indices, distances


def recurrent_weather_mixture_mae(
    root: Path, panel, dataset, candidates, indices, distances,
    checkpoint_path: Path, batch_size: int, device,
) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("future_weather_mode", "observed") != "observed":
        raise ValueError("Analog scenarios require an observed-weather recurrent checkpoint")
    if config.get("use_lagged_transport", True):
        raise ValueError("Analog probe requires the retrained no-lag checkpoint")
    model = TransportSourceRecurrentForecaster(
        root / "data/benchmarks/knowair/city.txt",
        stations=len(panel.stations), coordinates=panel.coordinates, **config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    weights = inverse_distance_weights(distances)
    steps = np.arange(24)
    total_error = total_count = 0.0
    offset = 0
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, 43, False):
            batch = move_batch(batch, device)
            size = batch["x"].shape[0]
            prediction = torch.zeros_like(batch["y"])
            for rank in range(indices.shape[1]):
                memory = candidates[indices[offset:offset + size, rank]]
                scenario = panel.values[memory[:, None] + steps[None], :, 1:]
                scenario_batch = dict(batch)
                scenario_batch["future_weather"] = torch.from_numpy(
                    np.asarray(scenario, dtype=np.float32)
                ).to(device)
                scenario_prediction = model(scenario_batch)["prediction"]
                weight = torch.as_tensor(
                    weights[offset:offset + size, rank], device=device,
                    dtype=scenario_prediction.dtype,
                )
                prediction += weight[:, None, None] * scenario_prediction
            physical_truth = batch["y"] * float(panel.std[0]) + float(panel.mean[0])
            valid = physical_truth >= 1e-4
            error = torch.abs(prediction - batch["y"]) * float(panel.std[0])
            total_error += float((error * valid).sum())
            total_count += float(valid.sum())
            offset += size
    return total_error / total_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/analog_memory/decision.json")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/retrained_ablation_no_lag/seed_43/best_model.pt",
    )
    parser.add_argument("--rolling-queries", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / args.output
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite one-shot probe: {output}")
    panel = load_panel(root)
    screen = rolling_screen(panel, args.rolling_queries)

    validation = CommonLocalWindowDataset(panel, "val")
    validation_origins = validation.starts + validation.history
    weather_choice = screen["selected_weather"]
    weather_candidates, weather_indices, weather_distances = retrieval(
        panel, validation_origins, weather_choice["key"], weather_choice["neighbors"]
    )
    weather_pm_mae = recurrent_weather_mixture_mae(
        root, panel, validation, weather_candidates, weather_indices, weather_distances,
        root / args.checkpoint, args.batch_size, choose_device(args.device),
    )

    pm_choice = screen["selected_pm_increment"]
    pm_candidates, pm_indices, pm_distances = retrieval(
        panel, validation_origins, pm_choice["key"], pm_choice["neighbors"]
    )
    pm_metrics = analog_continuation_metrics(
        panel.values, pm_candidates, validation_origins, pm_indices, pm_distances,
        float(panel.mean[0]), float(panel.std[0]),
    )
    weather_gain = V3_REFERENCE_MAE - weather_pm_mae
    pm_gain = V3_REFERENCE_MAE - pm_metrics["pm_increment_mae"]
    if weather_gain >= ADVANCEMENT_GAIN:
        decision = "advance weather-scenario memory to global/local neural prototypes"
    elif pm_gain >= ADVANCEMENT_GAIN:
        decision = "advance PM/source-template memory, not weather prediction"
    else:
        decision = "reject continuation memory; next probe is adaptive delayed-state retrieval"
    payload = {
        "protocol": {
            "memory_bank": "KnowAir train split only",
            "selection": "three expanding rolling-origin folds inside train",
            "original_validation_uses": 1,
            "checkpoint": args.checkpoint,
            "realized_future_weather_at_inference": False,
            "test_accessed": False,
        },
        "rolling_screen": screen,
        "original_validation": {
            "v3_reference_mae": V3_REFERENCE_MAE,
            "analog_weather_mixture_pm_mae": weather_pm_mae,
            "analog_weather_gain_vs_v3": weather_gain,
            "analog_pm_increment_mae": pm_metrics["pm_increment_mae"],
            "analog_pm_increment_gain_vs_v3": pm_gain,
            "pm_persistence_mae": pm_metrics["pm_persistence_mae"],
        },
        "development_advancement_gain": ADVANCEMENT_GAIN,
        "advance_memory_family": max(weather_gain, pm_gain) >= ADVANCEMENT_GAIN,
        "external_test_unlock_gate": "three-seed original-validation mean <= 18.0",
        "decision": decision,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
