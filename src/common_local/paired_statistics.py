"""Paired temporal uncertainty estimates for overlapping forecast origins."""

from __future__ import annotations

import numpy as np


def origin_mae(
    prediction: np.ndarray, truth: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return one masked MAE for every forecast origin.

    The validity rule is identical to :func:`common_local.metrics.physical_metrics`.
    Averaging is performed over horizons and stations within each origin so the
    resulting series can be resampled along its only independent axis: time.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must have the same [origin,horizon,station] shape")
    cleaned = truth.copy()
    cleaned[cleaned < 1e-4] = 0.0
    valid = cleaned != 0.0
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must have the same shape as truth")
        valid = valid & valid_mask
    counts = valid.sum(axis=(1, 2))
    if np.any(counts == 0):
        raise ValueError("At least one forecast origin has no valid target")
    absolute = np.abs(prediction - cleaned) * valid
    return absolute.sum(axis=(1, 2)) / counts


def circular_block_bootstrap(
    difference: np.ndarray,
    block_length: int,
    replicates: int = 10_000,
    seed: int = 20_260_812,
    chunk_size: int = 250,
) -> np.ndarray:
    """Bootstrap a mean with circular, fixed-length temporal blocks.

    Circular wrapping treats all observed origins symmetrically and avoids
    shortening the last sampled block. Results are generated in chunks to keep
    memory bounded for long validation sequences.
    """
    difference = np.asarray(difference, dtype=np.float64)
    if difference.ndim != 1 or len(difference) < 2:
        raise ValueError("difference must be a one-dimensional temporal series")
    if not 1 <= block_length <= len(difference):
        raise ValueError("block_length must lie between 1 and the series length")
    if replicates < 1 or chunk_size < 1:
        raise ValueError("replicates and chunk_size must be positive")
    rng = np.random.default_rng(seed)
    origins = len(difference)
    blocks = int(np.ceil(origins / block_length))
    offsets = np.arange(block_length, dtype=np.int64)
    estimates = np.empty(replicates, dtype=np.float64)
    for left in range(0, replicates, chunk_size):
        right = min(replicates, left + chunk_size)
        starts = rng.integers(0, origins, size=(right - left, blocks))
        indices = (starts[..., None] + offsets) % origins
        indices = indices.reshape(right - left, -1)[:, :origins]
        estimates[left:right] = difference[indices].mean(axis=1)
    return estimates


def summarize_paired_errors(
    reference_error: np.ndarray,
    proposed_error: np.ndarray,
    block_lengths: tuple[int, ...] = (56, 112, 224),
    replicates: int = 10_000,
    seed: int = 20_260_812,
) -> dict:
    """Summarize reference minus proposed origin-level MAE with block sensitivity."""
    reference_error = np.asarray(reference_error, dtype=np.float64)
    proposed_error = np.asarray(proposed_error, dtype=np.float64)
    if reference_error.shape != proposed_error.shape or reference_error.ndim != 1:
        raise ValueError("reference_error and proposed_error must be aligned one-dimensional arrays")
    difference = reference_error - proposed_error
    summaries = {}
    for index, length in enumerate(block_lengths):
        draws = circular_block_bootstrap(
            difference, length, replicates=replicates, seed=seed + index
        )
        lower, upper = np.quantile(draws, (0.025, 0.975))
        left_tail = (np.count_nonzero(draws <= 0.0) + 1) / (replicates + 1)
        right_tail = (np.count_nonzero(draws >= 0.0) + 1) / (replicates + 1)
        summaries[str(length)] = {
            "block_length_origins": int(length),
            "mean_mae_difference_reference_minus_proposed": float(difference.mean()),
            "ci95_percentile": [float(lower), float(upper)],
            "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(left_tail, right_tail))),
            "probability_difference_positive": float(np.mean(draws > 0.0)),
        }
    return {
        "origins": int(len(difference)),
        "reference_origin_mae": float(reference_error.mean()),
        "proposed_origin_mae": float(proposed_error.mean()),
        "mean_mae_difference_reference_minus_proposed": float(difference.mean()),
        "reference_better_origin_fraction": float(np.mean(difference < 0.0)),
        "proposed_better_origin_fraction": float(np.mean(difference > 0.0)),
        "block_sensitivity": summaries,
    }
