"""Physics-conditioned neighbor-by-lag features for residual diagnostics."""

from __future__ import annotations

import numpy as np


def neighbor_geometry(coordinates: np.ndarray, neighbors: int = 8):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    lon = np.radians(coordinates[:, 0]); lat = np.radians(coordinates[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    index = np.argsort(distance, axis=1)[:, 1:neighbors + 1]
    edge_distance = np.take_along_axis(distance, index, axis=1)
    source_lon, source_lat = lon[index], lat[index]
    east = lon[:, None] - source_lon
    east *= np.cos((lat[:, None] + source_lat) / 2)
    north = lat[:, None] - source_lat
    norm = np.hypot(east, north).clip(1e-8)
    return (
        index.astype(np.int64), edge_distance.astype(np.float32),
        (east / norm).astype(np.float32), (north / norm).astype(np.float32),
    )


def edge_time_features(
    history: np.ndarray, baseline_prediction: np.ndarray, coordinates: np.ndarray,
    feature_mean: np.ndarray, feature_std: np.ndarray, lags=(1, 2, 3, 4),
) -> np.ndarray:
    """Build conservative upstream-state contrasts for candidate edge×lag pairs.

    Scores combine geographic proximity, source-wind alignment and a soft
    travel-time mismatch. Features are zero-mean over target stations so a linear
    residual correction cannot masquerade as a source/sink term.
    """
    history = np.asarray(history, dtype=np.float32)
    prediction = np.asarray(baseline_prediction, dtype=np.float32)
    index, distance, east, north = neighbor_geometry(coordinates)
    features = []
    for lag in lags:
        source_pm = history[:, -lag, :, 0][:, index]
        wind_speed = (
            history[:, -lag, :, 4] * feature_std[4] + feature_mean[4]
        )[:, index]
        wind_sin = (
            history[:, -lag, :, 5] * feature_std[5] + feature_mean[5]
        )[:, index]
        wind_cos = (
            history[:, -lag, :, 6] * feature_std[6] + feature_mean[6]
        )[:, index]
        alignment = east[None] * (-wind_sin) + north[None] * (-wind_cos)
        travel_hours = distance[None] / np.maximum(wind_speed * 3.6, 1e-3)
        score = (
            -distance[None] / 300.0 + 2.0 * alignment
            - np.abs(travel_hours - lag * 3.0) / 6.0
        )
        score -= score.max(-1, keepdims=True)
        weight = np.exp(score)
        weight /= weight.sum(-1, keepdims=True).clip(1e-8)
        upstream = (weight * source_pm).sum(-1)
        contrast = upstream[:, None] - prediction
        contrast -= contrast.mean(-1, keepdims=True)
        features.append(contrast)
    return np.stack(features, -1).astype(np.float32)


def fit_horizon_ridge(
    features: np.ndarray, residual: np.ndarray, valid: np.ndarray, alpha: float,
) -> np.ndarray:
    horizon, feature_dim = features.shape[1], features.shape[-1]
    coefficients = np.zeros((horizon, feature_dim), dtype=np.float64)
    identity = np.eye(feature_dim)
    for step in range(horizon):
        mask = valid[:, step].reshape(-1)
        x = features[:, step].reshape(-1, feature_dim)[mask].astype(np.float64)
        y = residual[:, step].reshape(-1)[mask].astype(np.float64)
        gram = x.T @ x / max(len(x), 1)
        cross = x.T @ y / max(len(x), 1)
        coefficients[step] = np.linalg.solve(gram + alpha * identity, cross)
    return coefficients.astype(np.float32)


def corrected_mae(
    prediction: np.ndarray, truth: np.ndarray, features: np.ndarray,
    coefficients: np.ndarray, target_mean: float, target_std: float,
) -> tuple[float, float]:
    correction = np.einsum("bhnf,hf->bhn", features, coefficients, optimize=True)
    physical_truth = truth * target_std + target_mean
    valid = physical_truth >= 1e-4
    baseline = (np.abs(prediction - truth) * target_std * valid).sum() / valid.sum()
    corrected = (
        np.abs(prediction + correction - truth) * target_std * valid
    ).sum() / valid.sum()
    return float(baseline), float(corrected)
