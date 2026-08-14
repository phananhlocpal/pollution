#!/usr/bin/env python3
"""Test whether similar histories lead to divergent weather and PM futures.

Only train origins are used as candidate analogues and only validation origins
are queried. The test split is never constructed or read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common_local.data import CommonLocalWindowDataset, load_panel


def history_features(values: np.ndarray, starts: np.ndarray, history: int) -> np.ndarray:
    """Compact history while retaining temporal and station-level structure."""
    rows = []
    tendency_lag = min(8, history - 1)
    for start in starts:
        window = values[start:start + history]
        rows.append(np.concatenate((
            window.mean(1).ravel(),
            window.std(1).ravel(),
            window.mean(0).ravel(),
            window[-1].ravel(),
            (window[-1] - window[-1 - tendency_lag]).ravel(),
        )))
    return np.asarray(rows, dtype=np.float32)


def pair_divergence(panel, query_starts, reference_starts, history, horizon):
    values = panel.values
    count = len(query_starts)
    history_rmse = np.empty(count, dtype=np.float64)
    future_weather_rmse = np.empty(count, dtype=np.float64)
    future_pm_mae = np.empty(count, dtype=np.float64)
    weather_by_variable = np.empty((count, values.shape[-1] - 1), dtype=np.float64)
    for index, (query, reference) in enumerate(zip(query_starts, reference_starts)):
        query_history = values[query:query + history]
        reference_history = values[reference:reference + history]
        history_rmse[index] = np.sqrt(np.mean(
            np.square(query_history - reference_history), dtype=np.float64
        ))

        query_future = values[query + history:query + history + horizon]
        reference_future = values[
            reference + history:reference + history + horizon
        ]
        weather_difference = query_future[..., 1:] - reference_future[..., 1:]
        future_weather_rmse[index] = np.sqrt(np.mean(
            np.square(weather_difference), dtype=np.float64
        ))
        weather_by_variable[index] = np.sqrt(np.mean(
            np.square(weather_difference), axis=(0, 1), dtype=np.float64
        ))

        query_pm = query_future[..., 0] * panel.std[0] + panel.mean[0]
        reference_pm = reference_future[..., 0] * panel.std[0] + panel.mean[0]
        valid = (query_pm >= 1e-4) & (reference_pm >= 1e-4)
        future_pm_mae[index] = np.abs(query_pm - reference_pm)[valid].mean()
    return {
        "history_rmse_standardized": history_rmse,
        "future_weather_rmse_standardized": future_weather_rmse,
        "future_pm_divergence_mae": future_pm_mae,
        "future_weather_rmse_by_variable": weather_by_variable,
    }


def describe(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def summarize_subset(metrics, indices):
    return {
        "pairs": int(len(indices)),
        "history_rmse_standardized": describe(
            metrics["history_rmse_standardized"][indices]
        ),
        "future_weather_rmse_standardized": describe(
            metrics["future_weather_rmse_standardized"][indices]
        ),
        "future_pm_divergence_mae": describe(
            metrics["future_pm_divergence_mae"][indices]
        ),
    }


def history_matched_weather_contrast(metrics, pool_indices, bins=10):
    """Contrast weather divergence while balancing history distance by rank bins."""
    ordered = pool_indices[np.argsort(
        metrics["history_rmse_standardized"][pool_indices]
    )]
    low_groups, high_groups = [], []
    for group in np.array_split(ordered, bins):
        weather = metrics["future_weather_rmse_standardized"][group]
        low_cut, high_cut = np.quantile(weather, (0.25, 0.75))
        low_groups.append(group[weather <= low_cut])
        high_groups.append(group[weather >= high_cut])
    low = np.concatenate(low_groups)
    high = np.concatenate(high_groups)

    history_rank = rankdata(metrics["history_rmse_standardized"][pool_indices])
    weather_rank = rankdata(metrics["future_weather_rmse_standardized"][pool_indices])
    pm_rank = rankdata(metrics["future_pm_divergence_mae"][pool_indices])
    design = np.column_stack((np.ones(len(pool_indices)), history_rank))
    weather_residual = weather_rank - design @ np.linalg.lstsq(
        design, weather_rank, rcond=None
    )[0]
    pm_residual = pm_rank - design @ np.linalg.lstsq(
        design, pm_rank, rcond=None
    )[0]
    partial_spearman = np.corrcoef(weather_residual, pm_residual)[0, 1]
    low_pm = metrics["future_pm_divergence_mae"][low]
    high_pm = metrics["future_pm_divergence_mae"][high]
    return {
        "pool": "closest 25% of nearest-analogue pairs",
        "balancing": f"weather quartiles selected separately within {bins} history-distance bins",
        "low_weather_divergence": summarize_subset(metrics, low),
        "high_weather_divergence": summarize_subset(metrics, high),
        "pm_divergence_mean_delta": float(np.mean(high_pm) - np.mean(low_pm)),
        "pm_divergence_mean_ratio": float(np.mean(high_pm) / np.mean(low_pm)),
        "partial_spearman_controlling_history_distance": float(partial_spearman),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/history_future_divergence/analysis.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    panel = load_panel(root)
    train = CommonLocalWindowDataset(
        panel, "train", history=args.history, horizon=args.horizon
    )
    validation = CommonLocalWindowDataset(
        panel, "val", history=args.history, horizon=args.horizon
    )

    train_features = history_features(panel.values, train.starts, args.history)
    validation_features = history_features(
        panel.values, validation.starts, args.history
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    components = min(
        args.pca_components, train_scaled.shape[0] - 1, train_scaled.shape[1]
    )
    pca = PCA(
        n_components=components, whiten=True, svd_solver="randomized",
        random_state=args.seed,
    )
    train_embedding = pca.fit_transform(train_scaled)
    validation_embedding = pca.transform(validation_scaled)
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    neighbors.fit(train_embedding)
    embedding_distance, neighbor_index = neighbors.kneighbors(validation_embedding)
    matched_train_starts = train.starts[neighbor_index[:, 0]]
    matched = pair_divergence(
        panel, validation.starts, matched_train_starts, args.history, args.horizon
    )

    rng = np.random.default_rng(args.seed)
    random_train_starts = rng.choice(train.starts, size=len(validation), replace=True)
    random_pairs = pair_divergence(
        panel, validation.starts, random_train_starts, args.history, args.horizon
    )

    order = np.argsort(matched["history_rmse_standardized"])
    closest = {}
    for fraction in (0.01, 0.05, 0.10, 0.25, 1.0):
        size = max(1, int(round(len(order) * fraction)))
        closest[f"closest_{int(fraction * 100)}pct"] = summarize_subset(
            matched, order[:size]
        )

    close_indices = order[:max(4, int(round(len(order) * 0.10)))]
    close_weather = matched["future_weather_rmse_standardized"][close_indices]
    low_cut, high_cut = np.quantile(close_weather, (0.25, 0.75))
    low_indices = close_indices[close_weather <= low_cut]
    high_indices = close_indices[close_weather >= high_cut]
    low_pm = matched["future_pm_divergence_mae"][low_indices]
    high_pm = matched["future_pm_divergence_mae"][high_indices]
    correlation = spearmanr(
        matched["future_weather_rmse_standardized"][close_indices],
        matched["future_pm_divergence_mae"][close_indices],
    )

    variable_names = panel.feature_names[1:]
    closest_ten_variable = matched["future_weather_rmse_by_variable"][close_indices]
    result = {
        "analysis": "future divergence under equivalent histories",
        "splits": {
            "reference": "train",
            "query": "validation",
            "test_accessed": False,
        },
        "samples": {"train": len(train), "validation": len(validation)},
        "history_hours": args.history * panel.cadence_hours,
        "horizon_hours": args.horizon * panel.cadence_hours,
        "matching": {
            "features": (
                "regional lag means/stds plus station temporal means, latest "
                "state, and 24-hour tendency; train-standardized"
            ),
            "pca_components": components,
            "pca_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
            "embedding_nearest_neighbor_distance": describe(
                embedding_distance[:, 0]
            ),
        },
        "nearest_analogue_pairs": summarize_subset(matched, np.arange(len(validation))),
        "random_train_pairs": summarize_subset(
            random_pairs, np.arange(len(validation))
        ),
        "nearest_pairs_by_history_closeness": closest,
        "closest_10pct_weather_divergence_contrast": {
            "low_weather_divergence_quartile": summarize_subset(
                matched, low_indices
            ),
            "high_weather_divergence_quartile": summarize_subset(
                matched, high_indices
            ),
            "pm_divergence_mean_delta": float(np.mean(high_pm) - np.mean(low_pm)),
            "pm_divergence_mean_ratio": float(np.mean(high_pm) / np.mean(low_pm)),
            "weather_pm_spearman_r": float(correlation.statistic),
            "weather_pm_spearman_p": float(correlation.pvalue),
        },
        "history_distance_matched_weather_contrast": history_matched_weather_contrast(
            matched, order[:max(40, int(round(len(order) * 0.25)))]
        ),
        "closest_10pct_future_weather_rmse_by_variable": {
            name: describe(closest_ten_variable[:, index])
            for index, name in enumerate(variable_names)
        },
        "interpretation_guardrail": (
            "This is an analogue-based information diagnostic, not a formal "
            "Bayes-error lower bound or evidence that future PM causes weather."
        ),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
