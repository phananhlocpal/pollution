#!/usr/bin/env python3
"""Evaluate train-only spatial-mosaic and temporal-warp PM continuation oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analyze_history_future_divergence import history_features
from common_local.data import CommonLocalWindowDataset, load_panel


def spatial_regions(coordinates: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    if count == 1:
        return [np.arange(len(coordinates))]
    if count == len(coordinates):
        return [np.asarray([index]) for index in range(len(coordinates))]
    # Longitude degrees contract with latitude; correct it before Euclidean clustering.
    projected = coordinates.copy()
    projected[:, 0] *= np.cos(np.deg2rad(coordinates[:, 1].mean()))
    labels = KMeans(n_clusters=count, n_init=20, random_state=seed).fit_predict(projected)
    return [np.flatnonzero(labels == label) for label in range(count)]


def retrieve(values, train_starts, query_starts, stations, history, k, components, seed):
    train_features = history_features(values[:, stations], train_starts, history)
    query_features = history_features(values[:, stations], query_starts, history)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    query_scaled = scaler.transform(query_features)
    dimensions = min(components, len(train_scaled) - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=dimensions, whiten=True, svd_solver="randomized",
        random_state=seed,
    )
    train_embedding = pca.fit_transform(train_scaled)
    query_embedding = pca.transform(query_scaled)
    indices = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(
        train_embedding
    ).kneighbors(query_embedding, return_distance=False)
    return train_starts[indices], float(pca.explained_variance_ratio_.sum())


def region_oracle(
    physical, query_starts, donor_starts, stations, history, horizon, shifts,
    batch_size=32,
):
    """Return summed best-donor errors for absolute and residual continuations."""
    chunks = {mode: {"error_sum": 0.0, "valid_count": 0, "origin_mae": [], "best_shift": []}
              for mode in ("absolute", "residual")}
    for left in range(0, len(query_starts), batch_size):
        right = min(left + batch_size, len(query_starts))
        partial = _region_oracle_batch(
            physical, query_starts[left:right], donor_starts[left:right], stations,
            history, horizon, shifts,
        )
        for mode in chunks:
            chunks[mode]["error_sum"] += partial[mode]["error_sum"]
            chunks[mode]["valid_count"] += partial[mode]["valid_count"]
            chunks[mode]["origin_mae"].append(partial[mode]["origin_mae"])
            chunks[mode]["best_shift"].append(partial[mode]["best_shift"])
    for mode in chunks:
        chunks[mode]["origin_mae"] = np.concatenate(chunks[mode]["origin_mae"])
        chunks[mode]["best_shift"] = np.concatenate(chunks[mode]["best_shift"])
    return chunks


def _region_oracle_batch(
    physical, query_starts, donor_starts, stations, history, horizon, shifts,
):
    query_future = np.stack([
        physical[start + history:start + history + horizon, stations, 0]
        for start in query_starts
    ])
    query_last = np.stack([
        physical[start + history - 1, stations, 0] for start in query_starts
    ])
    valid = query_future >= 1e-4
    count = valid.sum((1, 2))
    modes = {"absolute": [], "residual": []}
    for shift in shifts:
        donor_future = np.stack([
            physical[
                start + history + shift:start + history + shift + horizon,
                stations, 0,
            ]
            for start in donor_starts.ravel()
        ]).reshape(*donor_starts.shape, horizon, len(stations))
        donor_last = np.stack([
            physical[start + history - 1, stations, 0]
            for start in donor_starts.ravel()
        ]).reshape(*donor_starts.shape, len(stations))
        candidate_valid = donor_future >= 1e-4
        shared_valid = candidate_valid & valid[:, None]
        denominator = shared_valid.sum((2, 3)).clip(min=1)
        absolute_error = (
            np.abs(donor_future - query_future[:, None]) * shared_valid
        ).sum((2, 3)) / denominator
        residual_prediction = query_last[:, None, None] + (
            donor_future - donor_last[:, :, None]
        )
        residual_error = (
            np.abs(residual_prediction - query_future[:, None]) * shared_valid
        ).sum((2, 3)) / denominator
        modes["absolute"].append(absolute_error)
        modes["residual"].append(residual_error)
    result = {}
    for mode, errors_by_shift in modes.items():
        errors = np.stack(errors_by_shift, axis=-1)
        flat = errors.reshape(len(query_starts), -1)
        best_flat = flat.argmin(1)
        best_error = flat[np.arange(len(flat)), best_flat]
        best_shift = np.asarray(shifts)[best_flat % len(shifts)]
        result[mode] = {
            "error_sum": float((best_error * count).sum()),
            "valid_count": int(count.sum()),
            "origin_mae": best_error,
            "best_shift": best_shift,
        }
    return result


def summarize(error_sum, valid_count, origins, shifts=None):
    row = {
        "mae": float(error_sum / valid_count),
        "mean_origin_mae": float(np.mean(origins)),
        "median_origin_mae": float(np.median(origins)),
    }
    if shifts is not None:
        row["selected_shift_steps"] = {
            str(int(value)): int((shifts == value).sum()) for value in np.unique(shifts)
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--regions", nargs="+", type=int, default=[1, 4, 8, 16, 32, 184])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--max-shift", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--output", default="artifacts/oracle_mosaic/validation.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    panel = load_panel(root)
    train = CommonLocalWindowDataset(panel, "train", history=args.history, horizon=args.horizon)
    validation = CommonLocalWindowDataset(
        panel, "val", max_samples=args.max_eval_samples,
        history=args.history, horizon=args.horizon,
    )
    # Ensure every shifted donor continuation remains wholly inside train.
    train_end = panel.split_points[0]
    eligible = train.starts[
        train.starts + args.history + args.horizon + args.max_shift <= train_end
    ]
    shifts = np.arange(-args.max_shift, args.max_shift + 1, dtype=int)
    results = {}
    for region_count in sorted(set(args.regions)):
        regions = spatial_regions(panel.coordinates, region_count, args.seed)
        totals = {
            key: {"error": 0.0, "count": 0, "origins": [], "shifts": []}
            for key in ("absolute_unwarped", "residual_unwarped", "absolute_warped", "residual_warped")
        }
        explained = []
        for region_index, stations in enumerate(regions):
            donors, variance = retrieve(
                panel.values, eligible, validation.starts, stations, args.history,
                args.k, args.pca_components, args.seed,
            )
            explained.append(variance)
            unwarped = region_oracle(
                panel.physical, validation.starts, donors, stations,
                args.history, args.horizon, [0],
            )
            warped = region_oracle(
                panel.physical, validation.starts, donors, stations,
                args.history, args.horizon, shifts,
            )
            for mode in ("absolute", "residual"):
                for suffix, source in (("unwarped", unwarped), ("warped", warped)):
                    key = f"{mode}_{suffix}"
                    row = source[mode]
                    totals[key]["error"] += row["error_sum"]
                    totals[key]["count"] += row["valid_count"]
                    totals[key]["origins"].append(row["origin_mae"])
                    if suffix == "warped":
                        totals[key]["shifts"].append(row["best_shift"])
            print(f"R={region_count} region={region_index + 1}/{len(regions)}", flush=True)
        result = {
            "region_sizes": [int(len(stations)) for stations in regions],
            "mean_pca_explained_variance_ratio": float(np.mean(explained)),
        }
        for key, total in totals.items():
            # Region-origin means are diagnostic only; global MAE is the primary metric.
            origins = np.concatenate(total["origins"])
            selected = np.concatenate(total["shifts"]) if total["shifts"] else None
            result[key] = summarize(total["error"], total["count"], origins, selected)
        results[str(region_count)] = result

    r16 = results.get("16")
    gate = None if r16 is None else {
        "residual_mosaic_unwarped_below_16_5": r16["residual_unwarped"]["mae"] < 16.5,
        "residual_mosaic_warped_below_15_5": r16["residual_warped"]["mae"] < 15.5,
    }
    output = {
        "analysis": "oracle spatial mosaic and temporal warp",
        "split": "validation",
        "test_accessed": False,
        "origins": len(validation),
        "retrieval": {
            "reference_split": "train", "k": args.k,
            "history_representation": "region-restricted core H1 history features",
            "pca_components": args.pca_components,
        },
        "temporal_warp_steps": shifts.tolist(),
        "temporal_warp_hours": (shifts * panel.cadence_hours).tolist(),
        "results": results,
        "r16_gate": gate,
        "guardrails": [
            "Candidates and continuations come from train only.",
            "Validation PM is used only to select the diagnostic oracle donor and shift.",
            "Absolute and residual atlases are reported separately; the historical 19.01 reference used TSR weather scenarios and is not a PM-patch baseline.",
        ],
    }
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
