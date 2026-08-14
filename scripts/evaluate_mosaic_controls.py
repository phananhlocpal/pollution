#!/usr/bin/env python3
"""Controls for deployability, locality, selection rank, and warp multiplicity."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analyze_history_future_divergence import history_features
from common_local.data import CommonLocalWindowDataset, load_panel
from evaluate_oracle_mosaic import region_oracle, spatial_regions


def retrieve(values, train_starts, query_starts, stations, history, k, components, seed):
    train_features = history_features(values[:, stations], train_starts, history)
    query_features = history_features(values[:, stations], query_starts, history)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    query_scaled = scaler.transform(query_features)
    dimensions = min(components, len(train_scaled) - 1, train_scaled.shape[1])
    pca = PCA(n_components=dimensions, whiten=True, svd_solver="randomized", random_state=seed)
    train_embedding = pca.fit_transform(train_scaled)
    query_embedding = pca.transform(query_scaled)
    distances, indices = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(
        train_embedding
    ).kneighbors(query_embedding)
    return train_starts[indices], distances


def weighted_median(values, weights, valid):
    weights = weights[:, :, None, None] * valid
    order = np.argsort(values, axis=1)
    ordered_values = np.take_along_axis(values, order, axis=1)
    ordered_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(ordered_weights, axis=1)
    threshold = ordered_weights.sum(1, keepdims=True) * .5
    index = (cumulative >= threshold).argmax(1)
    prediction = np.take_along_axis(ordered_values, index[:, None], axis=1)[:, 0]
    return prediction, threshold[:, 0] > 0


def deployable_region(
    physical, query_starts, donor_starts, distances, stations, history, horizon, ks,
    batch_size=32,
):
    totals = {
        f"{mode}_{method}_k{k}": [0.0, 0]
        for mode in ("absolute", "residual")
        for method in ("median", "weighted_median") for k in ks
    }
    for left in range(0, len(query_starts), batch_size):
        right = min(left + batch_size, len(query_starts))
        starts = query_starts[left:right]
        donors = donor_starts[left:right]
        truth = np.stack([physical[s + history:s + history + horizon, stations, 0] for s in starts])
        query_last = np.stack([physical[s + history - 1, stations, 0] for s in starts])
        donor_future = np.stack([
            physical[s + history:s + history + horizon, stations, 0] for s in donors.ravel()
        ]).reshape(len(starts), donors.shape[1], horizon, len(stations))
        donor_last = np.stack([physical[s + history - 1, stations, 0] for s in donors.ravel()]).reshape(
            len(starts), donors.shape[1], len(stations)
        )
        valid_truth = truth >= 1e-4
        valid_donor = donor_future >= 1e-4
        for k in ks:
            tau = np.median(distances[left:right, :k], axis=1, keepdims=True).clip(min=1e-6)
            weights = np.exp(-distances[left:right, :k] / tau)
            candidates = {
                "absolute": donor_future[:, :k],
                "residual": query_last[:, None, None] + donor_future[:, :k] - donor_last[:, :k, None],
            }
            for mode, candidate in candidates.items():
                masked = np.where(valid_donor[:, :k], candidate, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    median = np.nanmedian(masked, axis=1)
                fallback = np.broadcast_to(query_last[:, None], truth.shape)
                median = np.where(np.isfinite(median), median, fallback)
                wmedian, available = weighted_median(candidate, weights, valid_donor[:, :k])
                wmedian = np.where(available, wmedian, fallback)
                for method, prediction in (("median", median), ("weighted_median", wmedian)):
                    key = f"{mode}_{method}_k{k}"
                    totals[key][0] += float((np.abs(prediction - truth) * valid_truth).sum())
                    totals[key][1] += int(valid_truth.sum())
    return totals


def merge_totals(destination, source):
    for key, (error, count) in source.items():
        destination.setdefault(key, [0.0, 0])
        destination[key][0] += error
        destination[key][1] += count


def oracle_rank(
    physical, query_starts, donor_starts, stations, history, horizon, batch_size=32,
):
    ranks = []
    for left in range(0, len(query_starts), batch_size):
        right = min(left + batch_size, len(query_starts))
        starts, donors = query_starts[left:right], donor_starts[left:right]
        truth = np.stack([physical[s + history:s + history + horizon, stations, 0] for s in starts])
        last = np.stack([physical[s + history - 1, stations, 0] for s in starts])
        future = np.stack([physical[s + history:s + history + horizon, stations, 0] for s in donors.ravel()]).reshape(
            len(starts), donors.shape[1], horizon, len(stations)
        )
        donor_last = np.stack([physical[s + history - 1, stations, 0] for s in donors.ravel()]).reshape(
            len(starts), donors.shape[1], len(stations)
        )
        valid = (future >= 1e-4) & (truth[:, None] >= 1e-4)
        prediction = last[:, None, None] + future - donor_last[:, :, None]
        errors = (np.abs(prediction - truth[:, None]) * valid).sum((2, 3)) / valid.sum((2, 3)).clip(min=1)
        ranks.append(errors.argmin(1) + 1)
    return np.concatenate(ranks)


def random_regions(stations, count, seed):
    shuffled = np.random.default_rng(seed).permutation(stations)
    return [np.asarray(group) for group in np.array_split(shuffled, count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--regions", nargs="+", type=int, default=[16, 32, 46, 61, 64, 92, 128, 184])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/mosaic_controls/validation.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    panel = load_panel(root)
    train = CommonLocalWindowDataset(panel, "train", history=args.history, horizon=args.horizon)
    validation = CommonLocalWindowDataset(panel, "val", history=args.history, horizon=args.horizon)
    eligible = train.starts[train.starts + args.history + args.horizon + 4 <= panel.split_points[0]]
    max_k = max(args.ks)
    output_results = {}
    station_donors = []
    station_distances = []
    station_ranks = []
    for count in sorted(set(args.regions)):
        regions = spatial_regions(panel.coordinates, count, args.seed)
        deployable, oracle_error, oracle_count = {}, 0.0, 0
        def evaluate_geographic(index, stations):
            donors, distances = retrieve(panel.values, eligible, validation.starts, stations, args.history, max_k, args.pca_components, args.seed)
            local_deployable = deployable_region(
                panel.physical, validation.starts, donors, distances, stations,
                args.history, args.horizon, args.ks,
            )
            oracle = region_oracle(panel.physical, validation.starts, donors, stations, args.history, args.horizon, [0])
            rank = None
            if count == len(panel.stations):
                rank = oracle_rank(panel.physical, validation.starts, donors, stations, args.history, args.horizon)
            print(f"deployable R={count} region={index + 1}/{count}", flush=True)
            return local_deployable, oracle["residual"], donors, distances, rank
        rows = Parallel(n_jobs=4, prefer="threads")(
            delayed(evaluate_geographic)(index, stations)
            for index, stations in enumerate(regions)
        )
        for local_deployable, oracle, donors, distances, rank in rows:
            merge_totals(deployable, local_deployable)
            oracle_error += oracle["error_sum"]
            oracle_count += oracle["valid_count"]
            if count == len(panel.stations):
                station_donors.append(donors)
                station_distances.append(distances)
                station_ranks.append(rank)
        output_results[str(count)] = {
            "mean_stations_per_region": len(panel.stations) / count,
            "residual_oracle_mae_k100": oracle_error / oracle_count,
            "deployable_mae": {key: error / n for key, (error, n) in deployable.items()},
        }

    # Geographic-vs-random control at R=32, residual unwarped oracle only.
    random_control = {}
    for seed in args.random_seeds:
        error_sum = valid_count = 0
        def evaluate_random(index, stations):
            donors, _ = retrieve(panel.values, eligible, validation.starts, stations, args.history, max_k, args.pca_components, args.seed)
            row = region_oracle(panel.physical, validation.starts, donors, stations, args.history, args.horizon, [0])["residual"]
            print(f"random seed={seed} region={index + 1}/32", flush=True)
            return row
        random_rows = Parallel(n_jobs=4, prefer="threads")(
            delayed(evaluate_random)(index, stations)
            for index, stations in enumerate(random_regions(len(panel.stations), 32, seed))
        )
        for row in random_rows:
            error_sum += row["error_sum"]; valid_count += row["valid_count"]
        random_control[str(seed)] = error_sum / valid_count

    ranks = np.concatenate(station_ranks)
    rank_summary = {f"le_{cutoff}": float(np.mean(ranks <= cutoff)) for cutoff in (1, 3, 5, 10, 20, 50, 100)}
    rank_summary["mean"] = float(ranks.mean())
    rank_summary["median"] = float(np.median(ranks))

    # Equal-choice-count control: station-local K=900 unwarped vs K=100 x 9 shifts.
    k900_error = k900_count = warped_error = warped_count = 0
    def evaluate_choice(index, stations):
        donors900, _ = retrieve(panel.values, eligible, validation.starts, stations, args.history, 900, args.pca_components, args.seed)
        no_warp = region_oracle(panel.physical, validation.starts, donors900, stations, args.history, args.horizon, [0])["residual"]
        warped = region_oracle(panel.physical, validation.starts, station_donors[index], stations, args.history, args.horizon, range(-4, 5))["residual"]
        print(f"choice-control station={index + 1}/184", flush=True)
        return no_warp, warped
    choice_rows = Parallel(n_jobs=4, prefer="threads")(
        delayed(evaluate_choice)(index, stations)
        for index, stations in enumerate(spatial_regions(panel.coordinates, 184, args.seed))
    )
    for no_warp, warped in choice_rows:
        k900_error += no_warp["error_sum"]; k900_count += no_warp["valid_count"]
        warped_error += warped["error_sum"]; warped_count += warped["valid_count"]

    result = {
        "analysis": "mosaic deployability and confound controls",
        "split": "validation", "test_accessed": False,
        "origins": len(validation), "ks": args.ks,
        "geographic_results": output_results,
        "station_residual_oracle_history_rank": rank_summary,
        "geographic_vs_random_r32": {
            "geographic": output_results["32"]["residual_oracle_mae_k100"],
            "random_by_seed": random_control,
            "random_mean": float(np.mean(list(random_control.values()))),
        },
        "matched_choice_count": {
            "k900_unwarped_residual_oracle_mae": k900_error / k900_count,
            "k100_x9_window_shift_residual_oracle_mae": warped_error / warped_count,
            "note": "The shift changes the donor window; it is not an internal trajectory warp.",
        },
        "guardrails": [
            "All donors and continuations come from train only.",
            "Deployable medians use no validation future for selection.",
            "Oracle, rank, and matched-choice controls use validation truth for diagnosis only.",
        ],
    }
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
