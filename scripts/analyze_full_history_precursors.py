#!/usr/bin/env python3
"""Repeat the train-to-validation analogue diagnostic with H1--H4 histories.

The future-weather and future-PM outcomes are held fixed to the original core
panel.  Only the information used to retrieve a train analogue changes.  The
test split is neither constructed nor used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analyze_history_future_divergence import (
    describe,
    history_features,
    history_matched_weather_contrast,
    pair_divergence,
    summarize_subset,
)
from common_local.data import CommonLocalWindowDataset, RAW_FEATURES, load_panel


DIAGNOSTIC_FEATURES = (
    "surface_pressure_tendency_3h",
    "boundary_layer_height_tendency_3h",
    "wind_shear_950_minus_100m",
    "temperature_925_minus_950",
    "ventilation_100m",
)


def _train_standardize(values: np.ndarray, train_end: int) -> np.ndarray:
    train = values[:train_end].reshape(-1, values.shape[-1])
    mean = train.mean(0, dtype=np.float64)
    std = train.std(0, dtype=np.float64)
    if np.any(~np.isfinite(std)) or np.any(std < 1e-8):
        names = np.flatnonzero(std < 1e-8).tolist()
        raise ValueError(f"Degenerate history channels: {names}")
    return ((values - mean) / std).astype(np.float32)


def build_history_levels(root: Path, panel) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """Build leakage-free, train-standardized H1--H4 feature timelines."""
    raw = np.asarray(
        np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    train_end = panel.split_points[0]
    pm = raw[..., RAW_FEATURES.index("PM2.5")][..., None]
    raw_meteorology = raw[..., :-1]

    h1_physical = panel.physical
    auxiliary_physical = (
        panel.auxiliary * panel.auxiliary_std + panel.auxiliary_mean
    )
    h2_physical = np.concatenate((h1_physical, auxiliary_physical), axis=-1)
    h3_physical = np.concatenate((pm, raw_meteorology), axis=-1)

    def field(name: str) -> np.ndarray:
        return raw[..., RAW_FEATURES.index(name)]

    pressure = field("surface_pressure")
    boundary = field("boundary_layer_height")
    pressure_tendency = np.zeros_like(pressure)
    boundary_tendency = np.zeros_like(boundary)
    pressure_tendency[1:] = pressure[1:] - pressure[:-1]
    boundary_tendency[1:] = boundary[1:] - boundary[:-1]
    shear = np.hypot(
        field("u_wind_950") - field("100m_u_wind"),
        field("v_wind_950") - field("100m_v_wind"),
    )
    vertical_temperature = field("temperature_925") - field("temperature_950")
    ventilation = boundary * np.hypot(field("100m_u_wind"), field("100m_v_wind"))
    diagnostics = np.stack((
        pressure_tendency, boundary_tendency, shear, vertical_temperature,
        ventilation,
    ), axis=-1)
    h4_physical = np.concatenate((h3_physical, diagnostics), axis=-1)

    levels = {
        "H1_core": (h1_physical, panel.feature_names),
        "H2_core_auxiliary": (
            h2_physical, panel.feature_names + panel.auxiliary_feature_names,
        ),
        "H3_all_raw_meteorology": (
            h3_physical, ("PM2.5",) + RAW_FEATURES[:-1],
        ),
        "H4_raw_diagnostics": (
            h4_physical,
            ("PM2.5",) + RAW_FEATURES[:-1] + DIAGNOSTIC_FEATURES,
        ),
    }
    return {
        name: (_train_standardize(values, train_end), feature_names)
        for name, (values, feature_names) in levels.items()
    }


def evaluate_level(panel, train, validation, values, feature_names, args):
    train_compact = history_features(values, train.starts, args.history)
    validation_compact = history_features(values, validation.starts, args.history)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_compact)
    validation_scaled = scaler.transform(validation_compact)
    components = min(args.pca_components, len(train_scaled) - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=components, whiten=True, svd_solver="randomized",
        random_state=args.seed,
    )
    train_embedding = pca.fit_transform(train_scaled)
    validation_embedding = pca.transform(validation_scaled)
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(train_embedding)
    distance, indices = neighbors.kneighbors(validation_embedding)
    reference_starts = train.starts[indices[:, 0]]
    metrics = pair_divergence(
        panel, validation.starts, reference_starts, args.history, args.horizon,
        history_values=values,
    )
    order = np.argsort(metrics["history_rmse_standardized"])
    pool = order[:max(40, int(round(len(order) * .25)))]
    contrast = history_matched_weather_contrast(metrics, pool)
    close = order[:max(4, int(round(len(order) * .10)))]
    correlation = spearmanr(
        metrics["future_weather_rmse_standardized"][close],
        metrics["future_pm_divergence_mae"][close],
    )
    return {
        "history_channels": list(feature_names),
        "history_channel_count": len(feature_names),
        "pca_components": components,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "embedding_nearest_neighbor_distance": describe(distance[:, 0]),
        "nearest_analogue_pairs": summarize_subset(metrics, np.arange(len(validation))),
        "closest_10pct_pairs": summarize_subset(metrics, close),
        "closest_10pct_weather_pm_spearman_r": float(correlation.statistic),
        "closest_10pct_weather_pm_spearman_p": float(correlation.pvalue),
        "history_distance_matched_weather_contrast": contrast,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/full_history_precursors/validation.json"
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
    levels = build_history_levels(root, panel)
    results = {
        name: evaluate_level(panel, train, validation, values, names, args)
        for name, (values, names) in levels.items()
    }
    output = {
        "analysis": "history precursor information ladder H1-H4",
        "splits": {"reference": "train", "query": "validation", "test_accessed": False},
        "samples": {"train": len(train), "validation": len(validation)},
        "history_hours": args.history * panel.cadence_hours,
        "horizon_hours": args.horizon * panel.cadence_hours,
        "controlled_outcomes": (
            "All levels use the same core-six future-weather divergence and physical PM2.5 divergence; only analogue retrieval information changes."
        ),
        "levels": results,
        "interpretation_guardrail": (
            "Analogue retrieval is an information diagnostic, not a Bayes-error lower bound."
        ),
    }
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
