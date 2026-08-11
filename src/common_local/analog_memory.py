"""Leakage-safe historical analog retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RollingFold:
    candidate_origins: np.ndarray
    query_origins: np.ndarray


def rolling_origin_folds(
    train_end: int, history: int = 24, horizon: int = 24, folds: int = 3
) -> list[RollingFold]:
    """Create expanding folds whose memory values end before each dev block."""
    boundaries = np.linspace(train_end // 2, train_end, folds + 1, dtype=int)
    result = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        candidates = np.arange(history, left - horizon + 1, dtype=np.int64)
        queries = np.arange(left, right - horizon + 1, dtype=np.int64)
        if not len(candidates) or not len(queries):
            raise ValueError("Rolling fold is too short for history and horizon")
        result.append(RollingFold(candidates, queries))
    return result


def global_keys(
    values: np.ndarray, origins: np.ndarray, history: int, variant: str
) -> np.ndarray:
    """Build compact regional-regime keys from standardized past observations."""
    regional = np.asarray(values, dtype=np.float32).mean(1)
    if variant == "sequence":
        offsets = np.arange(history, 0, -1)
        return regional[origins[:, None] - offsets].reshape(len(origins), -1)
    if variant == "multiscale":
        lags = np.array((1, 2, 4, 8, 16, 24))
        if history < int(lags.max()):
            raise ValueError("History is shorter than multiscale key lags")
        samples = regional[origins[:, None] - lags]
        offsets = np.arange(history, 0, -1)
        windows = regional[origins[:, None] - offsets]
        return np.concatenate((
            samples.reshape(len(origins), -1), windows.mean(1), windows.std(1),
        ), 1)
    raise ValueError(f"Unknown analog key variant: {variant}")


def retrieve_neighbors(
    candidate_keys: np.ndarray, query_keys: np.ndarray, k: int, block_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact standardized Euclidean kNN indices and squared distances."""
    if k <= 0 or k > len(candidate_keys):
        raise ValueError("k must be positive and no larger than the memory bank")
    mean = candidate_keys.mean(0, dtype=np.float64)
    scale = candidate_keys.std(0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    candidates = ((candidate_keys - mean) / scale).astype(np.float32)
    candidate_norm = np.square(candidates).sum(1)
    all_indices, all_distances = [], []
    for left in range(0, len(query_keys), block_size):
        queries = ((query_keys[left:left + block_size] - mean) / scale).astype(np.float32)
        distance = (
            np.square(queries).sum(1, keepdims=True) + candidate_norm[None]
            - 2.0 * queries @ candidates.T
        )
        distance = np.maximum(distance, 0.0)
        partition = np.argpartition(distance, k - 1, axis=1)[:, :k]
        selected = np.take_along_axis(distance, partition, axis=1)
        order = np.argsort(selected, axis=1)
        all_indices.append(np.take_along_axis(partition, order, axis=1))
        all_distances.append(np.take_along_axis(selected, order, axis=1))
    return np.concatenate(all_indices), np.concatenate(all_distances)


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    distance = np.sqrt(np.maximum(distances, 0.0))
    weights = 1.0 / np.maximum(distance, 1e-4)
    exact = distance <= 1e-6
    has_exact = exact.any(1, keepdims=True)
    weights = np.where(has_exact, exact.astype(np.float64), weights)
    return weights / weights.sum(1, keepdims=True)


def analog_continuation_metrics(
    values: np.ndarray, candidate_origins: np.ndarray, query_origins: np.ndarray,
    neighbor_indices: np.ndarray, distances: np.ndarray, target_mean: float,
    target_std: float, horizon: int = 24, block_size: int = 16,
) -> dict:
    """Score weighted real-weather continuations and local PM increment templates."""
    weather_error2 = weather_count = 0.0
    weather_persistence_error2 = 0.0
    pm_error = pm_count = 0.0
    pm_persistence_error = 0.0
    weights = inverse_distance_weights(distances)
    steps = np.arange(horizon)
    for left in range(0, len(query_origins), block_size):
        right = min(left + block_size, len(query_origins))
        query = query_origins[left:right]
        memory = candidate_origins[neighbor_indices[left:right]]
        weight = weights[left:right]
        memory_time = memory[:, :, None] + steps[None, None]
        query_time = query[:, None] + steps[None]

        memory_pm = values[memory_time, :, 0]
        memory_last_pm = values[memory - 1, :, 0]
        increment = memory_pm - memory_last_pm[:, :, None]
        pm_prediction = values[query - 1, :, 0][:, None] + np.einsum(
            "bk,bkhn->bhn", weight, increment, optimize=True
        )
        pm_truth = values[query_time, :, 0]
        physical_truth = pm_truth * target_std + target_mean
        valid = physical_truth >= 1e-4
        pm_error += (np.abs(pm_prediction - pm_truth) * target_std * valid).sum()
        pm_persistence_error += (
            np.abs(values[query - 1, :, 0][:, None] - pm_truth) * target_std * valid
        ).sum()
        pm_count += valid.sum()

        memory_weather = values[memory_time, :, 1:]
        weather_prediction = np.einsum(
            "bk,bkhnf->bhnf", weight, memory_weather, optimize=True
        )
        weather_truth = values[query_time, :, 1:]
        weather_error2 += np.square(weather_prediction - weather_truth).sum()
        weather_persistence_error2 += np.square(
            values[query - 1, :, 1:][:, None] - weather_truth
        ).sum()
        weather_count += weather_truth.size
    return {
        "pm_increment_mae": float(pm_error / pm_count),
        "pm_persistence_mae": float(pm_persistence_error / pm_count),
        "weather_normalized_rmse": float(np.sqrt(weather_error2 / weather_count)),
        "weather_persistence_normalized_rmse": float(
            np.sqrt(weather_persistence_error2 / weather_count)
        ),
    }
