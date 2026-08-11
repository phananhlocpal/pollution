"""Residual EDA and cheap chronological probes for a frozen forecast bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common_local.data import RAW_FEATURES
from .evaluator import load_bundle


HORIZON_STEPS = (0, 1, 3, 7, 11, 15, 23)


def haversine_matrix(lat, lon):
    lat = np.deg2rad(lat); lon = np.deg2rad(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    value = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def neighbor_weights(coordinates, neighbors=8, scale_km=250.0):
    distance = haversine_matrix(coordinates[:, 1], coordinates[:, 0])
    np.fill_diagonal(distance, np.inf)
    nearest = np.argpartition(distance, neighbors, axis=1)[:, :neighbors]
    weights = np.zeros_like(distance)
    rows = np.arange(len(distance))[:, None]
    weights[rows, nearest] = np.exp(-distance[rows, nearest] / scale_km)
    return weights / weights.sum(1, keepdims=True)


def wind_aligned_innovation(raw, times, coordinates, base_weights):
    """Upwind-neighbour minus own PM using source-to-target wind alignment."""
    lat = np.deg2rad(coordinates[:, 1]); lon = np.deg2rad(coordinates[:, 0])
    north = lat[:, None] - lat[None, :]
    east = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
    norm = np.hypot(east, north)
    east = np.divide(east, norm, out=np.zeros_like(east), where=norm > 0)
    north = np.divide(north, norm, out=np.zeros_like(north), where=norm > 0)
    pm = np.asarray(raw[times, :, RAW_FEATURES.index("PM2.5")], dtype=np.float32)
    u = np.asarray(raw[times, :, RAW_FEATURES.index("100m_u_wind")], dtype=np.float32)
    v = np.asarray(raw[times, :, RAW_FEATURES.index("100m_v_wind")], dtype=np.float32)
    result = np.empty_like(pm)
    for index in range(len(times)):
        speed = np.hypot(u[index], v[index])
        unit_u = np.divide(u[index], speed, out=np.zeros_like(speed), where=speed > 1e-8)
        unit_v = np.divide(v[index], speed, out=np.zeros_like(speed), where=speed > 1e-8)
        alignment = np.maximum(0, east * unit_u[None, :] + north * unit_v[None, :])
        weights = base_weights * alignment
        total = weights.sum(1, keepdims=True)
        weights = np.divide(weights, total, out=base_weights.copy(), where=total > 1e-8)
        result[index] = weights @ pm[index] - pm[index]
    return result


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def fit_ridge(x_train, y_train, x_eval, alphas=(.01, .1, 1.0, 10.0, 100.0)):
    """Select alpha on the chronological tail of training, then refit."""
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train); evaluate = scaler.transform(x_eval)
    cut = int(len(train) * .8)
    best_alpha, best_mae = None, np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(train[:cut], y_train[:cut])
        mae = np.mean(np.abs(model.predict(train[cut:]) - y_train[cut:]))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    model = Ridge(alpha=best_alpha).fit(train, y_train)
    return model.predict(evaluate), best_alpha


def build_features(root, starts, coordinates):
    raw = np.load(Path(root) / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    pm_index = RAW_FEATURES.index("PM2.5")
    current = np.asarray(raw[starts - 1, :, pm_index], dtype=np.float32)
    previous = np.asarray(raw[starts - 2, :, pm_index], dtype=np.float32)
    weights = neighbor_weights(coordinates)
    spatial_innovation = current @ weights.T - current
    neighbor_trend = (current - previous) @ weights.T

    labels = KMeans(n_clusters=8, random_state=2026, n_init=20).fit_predict(coordinates)
    regional = np.empty_like(current)
    for label in np.unique(labels):
        mask = labels == label
        regional[:, mask] = current[:, mask].mean(1, keepdims=True) - current[:, mask]
    static = {
        "spatial_innovation": spatial_innovation,
        "neighbor_trend": neighbor_trend,
        "regional_factor": regional,
        "wind_aligned_innovation": wind_aligned_innovation(raw, starts - 1, coordinates, weights),
        "wind_lagged_innovation_3h": wind_aligned_innovation(raw, starts - 2, coordinates, weights),
        "wind_lagged_innovation_6h": wind_aligned_innovation(raw, starts - 3, coordinates, weights),
    }
    return raw, current, static, labels


def future_features(raw, times):
    def field(name):
        return np.asarray(raw[times, :, RAW_FEATURES.index(name)], dtype=np.float32)
    dew = field("2m_dewpoint"); temperature = field("2m_temperature")
    pbl = field("boundary_layer_height"); precip = field("total_precipitation")
    u = field("100m_u_wind"); v = field("100m_v_wind")
    return {
        "dewpoint": dew,
        "precipitation": precip,
        "boundary_layer_height": pbl,
        "ventilation": np.hypot(u, v) * pbl,
        "dewpoint_deficit": temperature - dew,
    }


def run(args):
    manifest, prediction, truth, starts = load_bundle(args.bundle)
    city = np.loadtxt(Path(args.root) / "data/benchmarks/knowair/city.txt", dtype=str)
    coordinates = city[:, 2:4].astype(float)
    if len(coordinates) != prediction.shape[2]:
        raise ValueError("city.txt order does not match bundle node count")
    raw, current, static, region_labels = build_features(args.root, starts, coordinates)
    split_origin = int(len(starts) * args.train_fraction)
    train_end = split_origin - 24
    if train_end <= 24 or len(starts) - split_origin <= 24:
        raise ValueError("Not enough origins for a chronological residual probe")

    signal_rows, probe_rows, station_rows, regime_rows = [], [], [], []
    for step in HORIZON_STEPS:
        hours = (step + 1) * 3
        residual = np.asarray(truth[:, step] - prediction[:, step])
        dynamic = future_features(raw, starts + step)
        signals = {**static, **dynamic}
        for name, values in signals.items():
            signal_rows.append({"horizon_hours": hours, "signal": name,
                                "residual_correlation": correlation(values.ravel(), residual.ravel())})

        base = np.stack((current, np.asarray(prediction[:, step])), axis=-1)
        y = residual
        train_rows = np.arange(len(starts)) < train_end
        eval_rows = np.arange(len(starts)) >= split_origin
        # Preserve chronology at origin level; flatten nodes only after splitting.
        base_train = base[train_rows].reshape(-1, 2)
        base_eval = base[eval_rows].reshape(-1, 2)
        y_train = y[train_rows].reshape(-1); y_eval = y[eval_rows].reshape(-1)
        base_correction, alpha = fit_ridge(base_train, y_train, base_eval)
        eval_prediction = np.asarray(prediction[eval_rows, step]).reshape(-1)
        eval_truth = np.asarray(truth[eval_rows, step]).reshape(-1)
        base_mae = float(np.mean(np.abs(eval_prediction - eval_truth)))
        corrected_mae = float(np.mean(np.abs(eval_prediction + base_correction - eval_truth)))
        probe_rows.append({"horizon_hours": hours, "signal": "baseline_current_pm+prediction",
                           "alpha": alpha, "base_mae": base_mae,
                           "corrected_mae": corrected_mae, "delta_mae": corrected_mae - base_mae})
        for name, values in signals.items():
            augmented = np.concatenate((base, values[..., None]), axis=-1)
            correction, alpha = fit_ridge(
                augmented[train_rows].reshape(-1, 3), y_train,
                augmented[eval_rows].reshape(-1, 3),
            )
            corrected = float(np.mean(np.abs(eval_prediction + correction - eval_truth)))
            probe_rows.append({"horizon_hours": hours, "signal": name, "alpha": alpha,
                               "base_mae": base_mae, "corrected_mae": corrected,
                               "delta_mae": corrected - base_mae,
                               "baseline_probe_mae": corrected_mae,
                               "incremental_delta_vs_baseline_probe": corrected - corrected_mae})

        absolute = np.abs(residual)
        for node, station in enumerate(city[:, 1]):
            station_rows.append({"horizon_hours": hours, "node": node, "station": station,
                                 "region": int(region_labels[node]),
                                 "mean_error": float(residual[:, node].mean()),
                                 "mae": float(absolute[:, node].mean())})
        thresholds = (0, 35, 75, 115, 150, 250, np.inf)
        target = np.asarray(truth[:, step])
        for low, high in zip(thresholds[:-1], thresholds[1:]):
            mask = (target >= low) & (target < high)
            regime_rows.append({"horizon_hours": hours, "pm_bin": f"[{low},{high})",
                                "points": int(mask.sum()),
                                "mae": float(absolute[mask].mean()) if mask.any() else None})

        target_times = starts + step
        timestamps = pd.Timestamp("2015-01-01") + pd.to_timedelta(target_times * 3, unit="h")
        months = np.asarray(timestamps.month)
        seasons = np.select(
            [np.isin(months, (12, 1, 2)), np.isin(months, (3, 4, 5)),
             np.isin(months, (6, 7, 8))], ["DJF", "MAM", "JJA"], default="SON"
        )
        for season in ("DJF", "MAM", "JJA", "SON"):
            mask = seasons == season
            regime_rows.append({"horizon_hours": hours, "season": season,
                                "points": int(mask.sum() * target.shape[1]),
                                "mae": float(absolute[mask].mean()) if mask.any() else None})

        train_limit = int(len(raw) * .5)
        p90 = np.quantile(np.asarray(raw[:train_limit, :, RAW_FEATURES.index("PM2.5")]), .9, axis=0)
        high_current = current >= p90[None, :]
        high_target = target >= p90[None, :]
        phases = {
            "normal": ~high_current & ~high_target,
            "onset": ~high_current & high_target,
            "continuation": high_current & high_target,
            "decay": high_current & ~high_target,
        }
        for phase, mask in phases.items():
            regime_rows.append({"horizon_hours": hours, "event_phase": phase,
                                "points": int(mask.sum()),
                                "mae": float(absolute[mask].mean()) if mask.any() else None})

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(signal_rows).to_csv(output / "residual_signal_correlations.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output / "chronological_ridge_probes.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output / "residual_by_station.csv", index=False)
    regimes = pd.DataFrame(regime_rows)
    regimes[regimes["pm_bin"].notna()].dropna(axis=1, how="all").to_csv(
        output / "residual_by_pm_bin.csv", index=False
    )
    regimes[regimes["season"].notna()].dropna(axis=1, how="all").to_csv(
        output / "residual_by_season.csv", index=False
    )
    regimes[regimes["event_phase"].notna()].dropna(axis=1, how="all").to_csv(
        output / "residual_by_event_phase.csv", index=False
    )
    summary = {
        "source_bundle": str(Path(args.bundle).resolve()), "source_manifest": manifest,
        "probe_split": {"train_origins": train_end,
                        "purged_gap_origins": 24,
                        "evaluation_origins": len(starts) - split_origin,
                        "inference_warning": "Screening only; final comparison needs paired block inference."},
        "selected_horizon_hours": [(step + 1) * 3 for step in HORIZON_STEPS],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/residual_probe")
    parser.add_argument("--train-fraction", type=float, default=.6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
