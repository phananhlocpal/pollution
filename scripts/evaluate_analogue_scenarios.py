#!/usr/bin/env python3
"""Evaluate history analogues as PM medians and future-weather TSR scenarios."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analyze_history_future_divergence import history_features
from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.train import choose_device


def origin_mae(prediction, truth, valid):
    error = np.abs(prediction - truth)
    return (error * valid).sum((1, 2)) / valid.sum((1, 2)).clip(min=1)


def update_global(accumulator, prediction, truth, valid):
    accumulator[0] += float((np.abs(prediction - truth) * valid).sum())
    accumulator[1] += int(valid.sum())


def future_stack(values, starts, history, horizon, columns):
    return np.stack([
        values[start + history:start + history + horizon, :, columns]
        for start in starts.ravel()
    ]).reshape(*starts.shape, horizon, values.shape[1], -1)


def load_models(root, seeds, pattern, panel, device):
    models = []
    for seed in seeds:
        checkpoint = torch.load(
            root / pattern.format(seed=seed), map_location=device,
            weights_only=False,
        )
        if checkpoint.get("config", {}).get("future_weather_mode") != "observed":
            raise ValueError("Scenario evaluation requires observed-weather TSR checkpoints")
        model = TransportSourceRecurrentForecaster(
            root / "data/benchmarks/knowair/city.txt",
            stations=len(panel.stations), **checkpoint["config"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, x, weather, device, batch_size):
    predictions = []
    with torch.inference_mode():
        for left in range(0, len(x), batch_size):
            right = min(left + batch_size, len(x))
            batch = {
                "x": torch.as_tensor(x[left:right], device=device),
                "future_weather": torch.as_tensor(
                    weather[left:right], device=device
                ),
            }
            total = None
            for model in models:
                value = model(batch)["prediction"]
                total = value if total is None else total + value
            predictions.append((total / len(models)).cpu().numpy())
    return np.concatenate(predictions)


def transport_dispersion(model, weather_scenarios):
    """RMS ensemble spread of normalized wind-aligned edge weights."""
    device = model.edge_weight.device
    weather = torch.as_tensor(weather_scenarios, device=device)
    wind_sin, wind_cos = weather[..., 4], weather[..., 5]
    flow_east = (-wind_sin[..., model.neighbor_index])
    flow_north = (-wind_cos[..., model.neighbor_index])
    alignment = torch.relu(
        model.edge_east[None, None] * flow_east
        + model.edge_north[None, None] * flow_north
    )
    weights = model.edge_weight[None, None] * alignment
    total = weights.sum(-1, keepdim=True)
    fallback = model.edge_weight / model.edge_weight.sum(-1, keepdim=True)
    weights = torch.where(
        total > 1e-8, weights / total.clamp_min(1e-8),
        fallback[None, None],
    )
    return float(weights.float().std(0, correction=0).square().mean().sqrt().cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--checkpoint-pattern",
        default="paper/checkpoints/tsr_primary/seed_{seed}.pt",
    )
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--scenario-batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/analogue_scenarios/validation.json"
    )
    args = parser.parse_args()
    ks = sorted(set(args.k))
    if ks[0] < 1:
        raise ValueError("K must be positive")

    root = Path(args.root).resolve()
    panel = load_panel(root)
    train = CommonLocalWindowDataset(
        panel, "train", history=args.history, horizon=args.horizon
    )
    validation = CommonLocalWindowDataset(
        panel, "val", args.max_eval_samples,
        history=args.history, horizon=args.horizon,
    )
    train_features = history_features(panel.values, train.starts, args.history)
    query_features = history_features(panel.values, validation.starts, args.history)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    query_scaled = scaler.transform(query_features)
    pca = PCA(
        n_components=args.pca_components, whiten=True, svd_solver="randomized",
        random_state=args.seed,
    )
    train_embedding = pca.fit_transform(train_scaled)
    query_embedding = pca.transform(query_scaled)
    retrieval = NearestNeighbors(n_neighbors=ks[-1], n_jobs=-1)
    retrieval.fit(train_embedding)
    distances, indices = retrieval.kneighbors(query_embedding)
    analogue_starts = train.starts[indices]

    models = load_models(
        root, args.seeds, args.checkpoint_pattern, panel,
        choose_device(args.device),
    )
    device = next(models[0].parameters()).device
    direct_global = {k: [0.0, 0] for k in ks}
    direct_best = {k: [] for k in ks}
    scenario_global = {k: [0.0, 0] for k in ks}
    scenario_best = {k: [] for k in ks}
    scenario_origin_error = {k: [] for k in ks}
    oracle_origin_error = []
    transport_spread = []
    weather_spread = []

    for left in range(0, len(validation), args.query_batch_size):
        right = min(left + args.query_batch_size, len(validation))
        query_starts = validation.starts[left:right]
        neighbor_starts = analogue_starts[left:right]
        batch_count = len(query_starts)
        truth_norm = np.stack([
            panel.values[
                start + args.history:start + args.history + args.horizon, :, 0
            ] for start in query_starts
        ])
        truth = truth_norm * panel.std[0] + panel.mean[0]
        valid = truth >= 1e-4

        analogue_pm_norm = future_stack(
            panel.values, neighbor_starts, args.history, args.horizon, [0]
        )[..., 0]
        analogue_pm = analogue_pm_norm * panel.std[0] + panel.mean[0]
        analogue_pm[analogue_pm < 1e-4] = np.nan
        for k in ks:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                direct_prediction = np.nanmedian(analogue_pm[:, :k], axis=1)
            fallback = np.stack([
                panel.values[start + args.history - 1, :, 0]
                for start in query_starts
            ])[:, None] * panel.std[0] + panel.mean[0]
            direct_prediction = np.where(
                np.isnan(direct_prediction), fallback, direct_prediction
            )
            update_global(direct_global[k], direct_prediction, truth, valid)
            candidate_valid = ~np.isnan(analogue_pm[:, :k]) & valid[:, None]
            candidate_error = (
                np.nan_to_num(np.abs(analogue_pm[:, :k] - truth[:, None]))
                * candidate_valid
            ).sum((2, 3)) / candidate_valid.sum((2, 3)).clip(min=1)
            direct_best[k].extend(candidate_error.min(1).tolist())

        analogue_weather = future_stack(
            panel.values, neighbor_starts, args.history, args.horizon,
            list(range(1, panel.values.shape[-1])),
        )
        query_history = np.stack([
            panel.values[start:start + args.history] for start in query_starts
        ])
        flat_history = np.repeat(query_history, ks[-1], axis=0)
        flat_weather = analogue_weather.reshape(
            batch_count * ks[-1], args.horizon, len(panel.stations), -1
        )
        scenario_prediction = ensemble_predict(
            models, flat_history, flat_weather, device, args.scenario_batch_size
        ).reshape(batch_count, ks[-1], args.horizon, len(panel.stations))
        scenario_prediction = scenario_prediction * panel.std[0] + panel.mean[0]
        for k in ks:
            median_prediction = np.median(scenario_prediction[:, :k], axis=1)
            update_global(scenario_global[k], median_prediction, truth, valid)
            median_error = origin_mae(median_prediction, truth, valid)
            scenario_origin_error[k].extend(median_error.tolist())
            candidate_error = np.stack([
                origin_mae(scenario_prediction[:, member], truth, valid)
                for member in range(k)
            ], axis=1)
            scenario_best[k].extend(candidate_error.min(1).tolist())

        true_weather = np.stack([
            panel.values[
                start + args.history:start + args.history + args.horizon, :, 1:
            ] for start in query_starts
        ])
        oracle_prediction = ensemble_predict(
            models, query_history, true_weather, device, args.scenario_batch_size
        ) * panel.std[0] + panel.mean[0]
        oracle_origin_error.extend(origin_mae(oracle_prediction, truth, valid).tolist())
        for row in range(batch_count):
            scenarios = analogue_weather[row]
            weather_spread.append(float(np.sqrt(np.square(scenarios.std(0)).mean())))
            transport_spread.append(transport_dispersion(models[0], scenarios))

        print(f"processed={right}/{len(validation)}", flush=True)

    oracle_error = np.asarray(oracle_origin_error)
    transport_spread = np.asarray(transport_spread)
    weather_spread = np.asarray(weather_spread)
    max_k_error = np.asarray(scenario_origin_error[ks[-1]])
    gap = max_k_error - oracle_error
    result = {
        "analysis": "K-analogue conditional median and TSR scenario oracle",
        "split": "validation",
        "test_accessed": False,
        "origins": len(validation),
        "retrieval": {
            "reference_split": "train",
            "pca_components": args.pca_components,
            "pca_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
            "mean_distance_by_k": {
                str(k): float(distances[:, k - 1].mean()) for k in ks
            },
        },
        "direct_analogue_pm": {
            str(k): {
                "conditional_median_mae": direct_global[k][0] / direct_global[k][1],
                "best_member_oracle_mean_origin_mae": float(np.mean(direct_best[k])),
            } for k in ks
        },
        "analogue_weather_tsr_ensemble": {
            str(k): {
                "scenario_median_mae": scenario_global[k][0] / scenario_global[k][1],
                "best_scenario_oracle_mean_origin_mae": float(np.mean(scenario_best[k])),
            } for k in ks
        },
        "true_future_weather_tsr_ensemble_mean_origin_mae": float(
            oracle_error.mean()
        ),
        "history_conditioned_forecastability": {
            "k": ks[-1],
            "transport_operator_spread_vs_scenario_error_spearman": float(
                spearmanr(transport_spread, max_k_error).statistic
            ),
            "transport_operator_spread_vs_oracle_gap_spearman": float(
                spearmanr(transport_spread, gap).statistic
            ),
            "weather_spread_vs_scenario_error_spearman": float(
                spearmanr(weather_spread, max_k_error).statistic
            ),
            "weather_spread_vs_oracle_gap_spearman": float(
                spearmanr(weather_spread, gap).statistic
            ),
        },
        "guardrails": [
            "All retrieval candidates and their futures come from train only.",
            "Best-of-K uses validation PM only as an oracle diagnostic, not for deployment.",
            "Scenario median and dispersion are deployable history-conditioned quantities.",
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
