"""Comparable, hypothesis-generating EDA across five air-quality benchmarks.

The diagnostics favour scale-free statistics and station-specific train
thresholds.  They are descriptive associations, not causal estimates.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, spearmanr

from .data import load_raw_frames


RNG = np.random.default_rng(42)
HORIZON_HOURS = (6, 24)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value))


def _rho(a: np.ndarray, b: np.ndarray, max_points: int = 300_000) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    indices = np.flatnonzero(valid)
    if len(indices) < 20:
        return np.nan
    if len(indices) > max_points:
        indices = RNG.choice(indices, max_points, replace=False)
    return float(spearmanr(a[indices], b[indices]).statistic)


def _partial_rank_rho(
    a: np.ndarray, b: np.ndarray, control: np.ndarray, max_points: int = 100_000
) -> float:
    """Partial Spearman association after linearly removing ranked control."""
    a, b, control = (np.asarray(value).reshape(-1) for value in (a, b, control))
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(control)
    indices = np.flatnonzero(valid)
    if len(indices) < 30:
        return np.nan
    if len(indices) > max_points:
        indices = RNG.choice(indices, max_points, replace=False)
    ar, br, cr = (rankdata(value[indices]) for value in (a, b, control))
    design = np.column_stack([np.ones(len(cr)), cr])
    a_residual = ar - design @ np.linalg.lstsq(design, ar, rcond=None)[0]
    b_residual = br - design @ np.linalg.lstsq(design, br, rcond=None)[0]
    return float(np.corrcoef(a_residual, b_residual)[0, 1])


def _run_lengths(flag: np.ndarray, value: bool) -> list[int]:
    selected = np.asarray(flag, dtype=bool) == value
    padded = np.r_[False, selected, False].astype(np.int8)
    changes = np.diff(padded)
    return (np.where(changes == -1)[0] - np.where(changes == 1)[0]).tolist()


def _haversine_distance(coordinates: np.ndarray) -> np.ndarray:
    lon = np.radians(coordinates[:, 0])
    lat = np.radians(coordinates[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(
        lat[None, :]
    ) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _knn(coordinates: np.ndarray, k: int) -> np.ndarray:
    distance = _haversine_distance(coordinates)
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1)[:, : min(k, len(coordinates) - 1)]


def _top_weight_neighbours(adjacency: np.ndarray, k: int) -> np.ndarray:
    weights = np.asarray(adjacency).copy()
    np.fill_diagonal(weights, -np.inf)
    return np.argsort(weights, axis=1)[:, -k:][:, ::-1]


def _neighbour_panel(panel: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    output = np.empty_like(panel, dtype=np.float32)
    for node, indices in enumerate(neighbours):
        with np.errstate(invalid="ignore"):
            output[:, node] = np.nanmean(panel[:, indices], axis=1)
    return output


def _future_max(panel: np.ndarray, steps: int) -> np.ndarray:
    result = np.full(panel.shape, np.nan, dtype=np.float32)
    for lead in range(1, steps + 1):
        shifted = np.full(panel.shape, np.nan, dtype=np.float32)
        shifted[:-lead] = panel[lead:]
        result = np.fmax(result, shifted)
    return result


def _panel_diagnostics(
    dataset: str,
    panel: np.ndarray,
    cadence_h: int,
    neighbours: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    scope: str,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    panel = np.asarray(panel, dtype=np.float32)
    t_count, n_nodes = panel.shape
    train_end = int(t_count * .70)
    test_start = int(t_count * .85)
    train = panel[:train_end]
    neighbour = _neighbour_panel(panel, neighbours)
    train_q25, train_q75 = np.nanquantile(train, [.25, .75], axis=0)
    scale = train_q75 - train_q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = np.nan
    threshold = np.nanquantile(train, .90, axis=0)

    observed = np.isfinite(panel)
    node_coverage = observed.mean(axis=0)
    values = panel[np.isfinite(panel)]
    overview = {
        "dataset": dataset,
        "scope": scope,
        "time_steps": t_count,
        "nodes": n_nodes,
        "cadence_h": cadence_h,
        "start": timestamps.min() if timestamps is not None else None,
        "end": timestamps.max() if timestamps is not None else None,
        "coverage_mean": float(observed.mean()),
        "coverage_node_p10": float(np.quantile(node_coverage, .10)),
        "coverage_node_p90": float(np.quantile(node_coverage, .90)),
        "pm_p50": float(np.quantile(values, .50)),
        "pm_p90": float(np.quantile(values, .90)),
        "pm_p99": float(np.quantile(values, .99)),
        "tail_p99_over_p90": float(np.quantile(values, .99) / np.quantile(values, .90)),
    }

    horizon_rows = []
    native_steps = sorted(set([1] + [h // cadence_h for h in HORIZON_HOURS if h % cadence_h == 0]))
    recent = panel[1:] - panel[:-1]
    neighbour_recent = neighbour[1:] - neighbour[:-1]
    for step in native_steps:
        if step <= 0 or step >= t_count:
            continue
        horizon = step * cadence_h
        current = panel[:-step]
        future = panel[step:]
        delta = future - current
        aligned_recent = recent[: t_count - step - 1]
        aligned_neighbour_recent = neighbour_recent[: t_count - step - 1]
        aligned_delta = delta[1:]
        gap = neighbour[:-step] - current

        node_rows = []
        for node in range(n_nodes):
            node_rows.append(
                (
                    _rho(aligned_recent[:, node], aligned_delta[:, node], 100_000),
                    _rho(aligned_neighbour_recent[:, node], aligned_delta[:, node], 100_000),
                    _rho(gap[:, node], delta[:, node], 100_000),
                    _partial_rank_rho(
                        gap[:, node], delta[:, node], current[:, node], 100_000
                    ),
                )
            )
        node_rho = np.asarray(node_rows)
        normalized_error = np.abs(delta) / scale[None, :]
        city_delta = np.nanmean(delta, axis=1)
        local_delta = delta - city_delta[:, None]
        regional_var = np.nanvar(city_delta)
        local_var = np.nanvar(local_delta)
        horizon_rows.append(
            {
                "dataset": dataset,
                "horizon_h": horizon,
                "persistence_rho": _rho(current, future),
                "persistence_mae_over_train_iqr": float(np.nanmedian(normalized_error)),
                "current_level_vs_future_change_rho": _rho(current, delta),
                "own_recent_trend_rho_median_node": float(np.nanmedian(node_rho[:, 0])),
                "neighbour_recent_trend_rho_median_node": float(np.nanmedian(node_rho[:, 1])),
                "spatial_gap_rho_median_node": float(np.nanmedian(node_rho[:, 2])),
                "spatial_gap_partial_current_rho_median_node": float(
                    np.nanmedian(node_rho[:, 3])
                ),
                "cross_node_mean_component_share": float(
                    regional_var / (regional_var + local_var)
                ),
            }
        )

    # Edge lead-lag of native-step changes.  Restrict to a deterministic edge
    # sample so nationwide/global panels remain tractable.
    edges = [(source, target) for target in range(n_nodes) for source in neighbours[target]]
    if len(edges) > 400:
        choice = RNG.choice(len(edges), 400, replace=False)
        edges = [edges[index] for index in choice]
    lag_steps = sorted(set([0, 1] + [h // cadence_h for h in (6, 12) if h % cadence_h == 0]))
    edge_rows = []
    best_zero = []
    change = panel[1:] - panel[:-1]
    for source, target in edges:
        correlations = {}
        for lag in lag_steps:
            if lag == 0:
                rho = _rho(change[:, source], change[:, target], 100_000)
            else:
                rho = _rho(change[:-lag, source], change[lag:, target], 100_000)
            correlations[lag] = rho
            edge_rows.append(
                {
                    "dataset": dataset,
                    "lag_h": lag * cadence_h,
                    "source": int(source),
                    "target": int(target),
                    "rho": rho,
                }
            )
        finite = {lag: rho for lag, rho in correlations.items() if np.isfinite(rho)}
        if finite:
            best_zero.append(max(finite, key=lambda lag: abs(finite[lag])) == 0)

    # Event and data-quality diagnostics use station-specific train p90.
    exceed = panel >= threshold[None, :]
    test = slice(test_start, None)
    test_exceed = exceed[test] & observed[test]
    test_observed = observed[test]
    active_fraction = test_exceed.sum(axis=1) / np.maximum(test_observed.sum(axis=1), 1)
    active_fraction = active_fraction[active_fraction > 0]
    neighbour_exceedance = _neighbour_panel(
        np.where(test_observed, test_exceed, np.nan).astype(np.float32), neighbours
    )
    local_coherence = neighbour_exceedance[test_exceed]
    base_event_rate = test_exceed.sum() / max(test_observed.sum(), 1)
    node_event_rate = test_exceed.sum(axis=0) / np.maximum(test_observed.sum(axis=0), 1)
    expected_by_target = np.asarray([
        np.mean(node_event_rate[indices]) for indices in neighbours
    ])
    target_event_count = test_exceed.sum(axis=0)
    expected_local_coherence = float(
        np.average(expected_by_target, weights=np.maximum(target_event_count, 1))
    )

    spike_lengths = []
    missing_lengths = []
    for node in range(n_nodes):
        spike_lengths.extend(_run_lengths(test_exceed[:, node], True))
        missing_lengths.extend(_run_lengths(observed[:, node], False))
    spike_hours = np.asarray(spike_lengths, dtype=float) * cadence_h
    missing_hours = np.asarray(missing_lengths, dtype=float) * cadence_h

    max_24 = _future_max(panel, max(1, 24 // cadence_h))
    ratio = panel / threshold[None, :]
    eligible = observed & np.isfinite(max_24) & (panel < threshold[None, :])
    onset = max_24 >= threshold[None, :]
    onset_rows = []
    for low, high in zip((0, .25, .50, .75), (.25, .50, .75, 1.0)):
        mask = eligible & (ratio >= low) & (ratio < high)
        onset_rows.append(
            {
                "dataset": dataset,
                "current_over_p90_bin": f"{low:.2f}-{high:.2f}",
                "n": int(mask.sum()),
                "onset_24h_rate": float(onset[mask].mean()) if mask.any() else np.nan,
            }
        )

    monthly_amplitude = np.nan
    diurnal_amplitude = np.nan
    weekend_minus_weekday = np.nan
    if timestamps is not None:
        temporal_level = pd.Series(np.nanmedian(panel, axis=1), index=timestamps)
        monthly = temporal_level.groupby(
            timestamps.month
        ).median()
        global_iqr = np.nanquantile(panel, .75) - np.nanquantile(panel, .25)
        monthly_amplitude = float((monthly.max() - monthly.min()) / global_iqr)
        diurnal = temporal_level.groupby(timestamps.hour).median()
        diurnal_amplitude = float((diurnal.max() - diurnal.min()) / global_iqr)
        weekend_minus_weekday = float(
            (temporal_level[timestamps.dayofweek >= 5].median()
             - temporal_level[timestamps.dayofweek < 5].median()) / global_iqr
        )

    train_values = train[np.isfinite(train)]
    test_values = panel[test][np.isfinite(panel[test])]
    event_rows = [{
        "dataset": dataset,
        "event_rate_test": base_event_rate,
        "local_event_coherence": float(np.nanmean(local_coherence)),
        "local_event_expected_if_time_independent": expected_local_coherence,
        "local_event_lift_over_independent": float(
            np.nanmean(local_coherence) / expected_local_coherence
        ),
        "active_node_fraction_median": float(np.nanmedian(active_fraction)),
        "active_node_fraction_p90": float(np.nanquantile(active_fraction, .90)),
        "spike_duration_h_median": float(np.nanmedian(spike_hours)),
        "spike_duration_h_p90": float(np.nanquantile(spike_hours, .90)),
        "coverage_mean": float(observed.mean()),
        "missing_gap_h_p90": float(np.nanquantile(missing_hours, .90)) if len(missing_hours) else 0,
        "missing_gap_h_max": float(np.nanmax(missing_hours)) if len(missing_hours) else 0,
        "timestamps_10pct_nodes_missing_rate": float((1 - observed.mean(axis=1) >= .10).mean()),
        "seasonal_amplitude_over_iqr": monthly_amplitude,
        "diurnal_amplitude_over_iqr": diurnal_amplitude,
        "weekend_minus_weekday_over_iqr": weekend_minus_weekday,
        "node_median_spread_over_global_iqr": float(
            np.nanstd(np.nanmedian(panel, axis=0))
            / (np.nanquantile(panel, .75) - np.nanquantile(panel, .25))
        ),
        "native_change_up_p99_over_down_p99": float(
            np.nanquantile(np.diff(panel, axis=0), .99)
            / -np.nanquantile(np.diff(panel, axis=0), .01)
        ),
        "test_minus_train_median_over_train_iqr": float(
            (np.median(test_values) - np.median(train_values))
            / (np.quantile(train_values, .75) - np.quantile(train_values, .25))
        ),
    }]

    edge_summary = (
        pd.DataFrame(edge_rows).groupby(["dataset", "lag_h"]).rho.agg(
            edge_count="count", median_rho="median", mean_rho="mean"
        ).reset_index().to_dict(orient="records")
    )
    for row in edge_summary:
        row["best_lag_zero_edge_fraction"] = float(np.mean(best_zero))
    return overview, horizon_rows, event_rows + onset_rows, edge_summary


def _horizon_regime_diagnostics(
    dataset: str,
    panel: np.ndarray,
    cadence_h: int,
    neighbours: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> list[dict]:
    """Re-estimate key associations within chronological and seasonal strata."""
    panel = np.asarray(panel, dtype=np.float32)
    neighbour = _neighbour_panel(panel, neighbours)
    total = len(panel)
    train_end, val_end = int(total * .70), int(total * .85)
    split = np.select(
        [np.arange(total) < train_end, np.arange(total) < val_end],
        ["train", "val"], default="test",
    )
    season = pd.Series(timestamps.month).map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }).to_numpy()
    rows = []
    for horizon in HORIZON_HOURS:
        if horizon % cadence_h:
            continue
        step = horizon // cadence_h
        current, future = panel[:-step], panel[step:]
        delta = future - current
        gap = neighbour[:-step] - current
        own_recent = panel[1:len(panel)-step] - panel[:len(panel)-step-1]
        neighbour_recent = neighbour[1:len(panel)-step] - neighbour[:len(panel)-step-1]
        # Recent arrays and delta[1:] correspond to current times 1..T-step-1.
        strata = {
            **{f"split:{value}": split[:-step] == value for value in ("train", "val", "test")},
            **{f"season:{value}": season[:-step] == value for value in ("winter", "spring", "summer", "autumn")},
        }
        for stratum, mask in strata.items():
            recent_mask = mask[1:]
            rows.append({
                "dataset": dataset,
                "horizon_h": horizon,
                "stratum": stratum,
                "n_timestamps": int(mask.sum()),
                "persistence_rho": _rho(current[mask], future[mask]),
                "current_level_vs_future_change_rho": _rho(current[mask], delta[mask]),
                "spatial_gap_rho": _rho(gap[mask], delta[mask]),
                "spatial_gap_partial_current_rho": _partial_rank_rho(
                    gap[mask], delta[mask], current[mask]
                ),
                "own_recent_trend_rho": _rho(own_recent[recent_mask], delta[1:][recent_mask]),
                "neighbour_recent_trend_rho": _rho(
                    neighbour_recent[recent_mask], delta[1:][recent_mask]
                ),
            })
    return rows


def _distance_diagnostics(
    dataset: str, panel: np.ndarray, coordinates: np.ndarray, max_pairs: int = 1_000
) -> tuple[list[dict], dict]:
    distance = _haversine_distance(coordinates)
    pairs = list(zip(*np.triu_indices(panel.shape[1], 1)))
    if len(pairs) > max_pairs:
        pairs = [pairs[index] for index in RNG.choice(len(pairs), max_pairs, replace=False)]
    change = np.diff(panel, axis=0)
    rows = []
    for a, b in pairs:
        rows.append({
            "dataset": dataset,
            "node_a": int(a), "node_b": int(b),
            "distance_km": float(distance[a, b]),
            "level_rho": _rho(panel[:, a], panel[:, b], 100_000),
            "native_change_rho": _rho(change[:, a], change[:, b], 100_000),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "dataset": dataset,
        "pair_count": len(frame),
        "distance_vs_level_rho": _rho(frame.distance_km, frame.level_rho),
        "distance_vs_native_change_rho": _rho(
            frame.distance_km, frame.native_change_rho
        ),
    }
    return rows, summary


def _knowair_wind_alignment(
    root: Path, neighbours: np.ndarray, coordinates: np.ndarray
) -> tuple[list[dict], list[dict]]:
    """Replicate the UCI directed wind/edge test at KnowAir's 3-hour cadence."""
    array = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    pm = np.asarray(array[:, :, -1], dtype=np.float32)
    u = np.asarray(array[:, :, 13], dtype=np.float32)  # 950 hPa eastward wind
    v = np.asarray(array[:, :, 14], dtype=np.float32)  # 950 hPa northward wind
    timestamps = pd.date_range("2015-01-01", periods=len(pm), freq="3h")
    change = np.diff(pm, axis=0)
    lon, lat = np.radians(coordinates[:, 0]), np.radians(coordinates[:, 1])
    edges = [(source, target) for target in range(pm.shape[1]) for source in neighbours[target]]
    edges = [edges[index] for index in RNG.choice(len(edges), min(400, len(edges)), replace=False)]
    pair_rows, robust_rows = [], []
    split = np.select(
        [np.arange(len(change)) < int(len(change) * .70), np.arange(len(change)) < int(len(change) * .85)],
        ["train", "val"], default="test",
    )
    season = pd.Series(timestamps[1:].month).map({
        12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn",
    }).to_numpy()
    for source, target in edges:
        dx = (lon[target] - lon[source]) * np.cos((lat[target] + lat[source]) / 2)
        dy = lat[target] - lat[source]
        norm = np.hypot(dx, dy)
        east, north = dx / norm, dy / norm
        speed = np.hypot(u[1:, source], v[1:, source])
        alignment = (u[1:, source] * east + v[1:, source] * north) / np.maximum(speed, 1e-6)
        for lag_step in (0, 1, 2, 4):
            if lag_step:
                source_change = change[:-lag_step, source]
                target_change = change[lag_step:, target]
                aligned_value = alignment[:-lag_step]
                wind_speed = speed[:-lag_step]
            else:
                source_change = change[:, source]
                target_change = change[:, target]
                aligned_value, wind_speed = alignment, speed
            correlations = {}
            counts = {}
            for regime, mask in {
                "aligned": (wind_speed >= 1) & (aligned_value >= .5),
                "opposed": (wind_speed >= 1) & (aligned_value <= -.5),
            }.items():
                correlations[regime] = _rho(source_change[mask], target_change[mask])
                counts[regime] = int(mask.sum())
            pair_rows.append({
                "dataset": "KnowAir", "source": source, "target": target,
                "lag_h": lag_step * 3,
                "aligned_n": counts["aligned"], "opposed_n": counts["opposed"],
                "aligned_rho": correlations["aligned"],
                "opposed_rho": correlations["opposed"],
                "aligned_minus_opposed": correlations["aligned"] - correlations["opposed"],
            })

        # Robustness of the first resolvable transport lag (+3 h).
        source_change, target_change = change[:-1, source], change[1:, target]
        aligned_value, wind_speed = alignment[:-1], speed[:-1]
        strata = {
            **{f"split:{value}": split[:-1] == value for value in ("train", "val", "test")},
            **{f"season:{value}": season[:-1] == value for value in ("winter", "spring", "summer", "autumn")},
            "speed:moderate": (wind_speed >= 1) & (wind_speed < 5),
            "speed:strong": wind_speed >= 5,
        }
        for stratum, base_mask in strata.items():
            values = {}
            for regime, direction_mask in {
                "aligned": aligned_value >= .5, "opposed": aligned_value <= -.5,
            }.items():
                mask = base_mask & (wind_speed >= 1) & direction_mask
                values[regime] = _rho(source_change[mask], target_change[mask])
            robust_rows.append({
                "dataset": "KnowAir", "source": source, "target": target,
                "stratum": stratum,
                "aligned_minus_opposed": values["aligned"] - values["opposed"],
            })
    return pair_rows, robust_rows


def _load_uci(root: Path):
    frame = load_raw_frames(root / "data/raw/PRSA_Data_20130301-20170228")
    stations = sorted(frame.station.unique())
    wide = frame.pivot(index="timestamp", columns="station", values="PM2.5").reindex(columns=stations)
    coords = pd.read_csv(root / "data/metadata/uci_beijing_station_coords.csv").set_index("station")
    xy = coords.loc[stations, ["longitude", "latitude"]].to_numpy()
    return wide.to_numpy(np.float32), wide.index, _knn(xy, 4), xy


def _parse_kdd(path: Path) -> dict[str, pd.DataFrame]:
    records = []
    in_data = False
    for raw in path.open():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        _, city, station, pollutant, start, raw_values = line.split(":", 5)
        if pollutant != "PM2.5":
            continue
        values = np.asarray(
            [np.nan if value == "?" else float(value) for value in raw_values.split(",")],
            dtype=np.float32,
        )
        records.append((city, station, pd.to_datetime(start), values))
    output = {}
    for city in sorted({row[0] for row in records}):
        series = []
        for _, station, start, values in (row for row in records if row[0] == city):
            series.append(pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), name=station))
        output[city] = pd.concat(series, axis=1).sort_index()
    return output


def _load_knowair(root: Path):
    array = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    panel = np.asarray(array[:, :, -1], dtype=np.float32)
    cities = pd.read_csv(
        root / "data/benchmarks/knowair/city.txt", sep=r"\s+", header=None,
        names=["id", "city", "lon", "lat"],
    )
    timestamps = pd.date_range("2015-01-01", periods=len(panel), freq="3h")
    coordinates = cities[["lon", "lat"]].to_numpy()
    return panel, timestamps, _knn(coordinates, 10), coordinates


def _load_airqualitybench(root: Path, n_nodes: int = 250):
    base = root / "data/benchmarks/airqualitybench"
    total_observed = np.zeros(3720, dtype=np.int64)
    total_hours = 0
    for path in sorted(base.glob("aq_compact_*.h5")):
        with h5py.File(path) as handle:
            total_observed += handle["masks"][:, :, 0].sum(axis=0)
            total_hours += handle["masks"].shape[0]
    selected = np.sort(np.argsort(total_observed / total_hours)[-n_nodes:])
    panels, dates = [], []
    for path in sorted(base.glob("aq_compact_*.h5")):
        year = int(path.stem.rsplit("_", 1)[-1])
        with h5py.File(path) as handle:
            values = handle["values"][:, selected, 0].astype(np.float32)
            mask = handle["masks"][:, selected, 0].astype(bool)
        values[~mask] = np.nan
        panels.append(values)
        dates.append(pd.date_range(f"{year}-01-01", periods=len(values), freq="h"))
    metadata = pd.read_csv(base / "selected_nodes_metadata.csv").iloc[selected]
    coordinates = metadata[["lon", "lat"]].to_numpy()
    return (
        np.concatenate(panels), dates[0].append(dates[1:]),
        _knn(coordinates, 10), coordinates, selected,
    )


def _coupling_metrics(
    dataset: str,
    pm: np.ndarray,
    other: np.ndarray,
    pollutant: str,
    cadence_h: int,
) -> list[dict]:
    rows = []
    pm_change = np.diff(pm, axis=0)
    other_change = np.diff(other, axis=0)
    level_rho, change_rho = [], []
    for node in range(pm.shape[1]):
        level_rho.append(_rho(pm[:, node], other[:, node]))
        change_rho.append(_rho(pm_change[:, node], other_change[:, node]))
    for horizon in HORIZON_HOURS:
        if horizon % cadence_h:
            continue
        step = horizon // cadence_h
        future_delta = pm[step:] - pm[:-step]
        precursor, partial = [], []
        for node in range(pm.shape[1]):
            precursor.append(_rho(other[:-step, node], future_delta[:, node]))
            partial.append(_partial_rank_rho(
                other[:-step, node], future_delta[:, node], pm[:-step, node]
            ))
        rows.append({
            "dataset": dataset, "pollutant": pollutant, "horizon_h": horizon,
            "level_rho_median_node": float(np.nanmedian(level_rho)),
            "native_change_rho_median_node": float(np.nanmedian(change_rho)),
            "current_pollutant_vs_future_pm_change_rho_median_node": float(np.nanmedian(precursor)),
            "partial_current_pm_rho_median_node": float(np.nanmedian(partial)),
        })
    return rows


def _pollutant_coupling(root: Path, aq_selected: np.ndarray) -> list[dict]:
    output = []
    uci = load_raw_frames(root / "data/raw/PRSA_Data_20130301-20170228").sort_values(
        ["station", "timestamp"]
    )
    stations = sorted(uci.station.unique())
    uci_panels = {
        feature: uci.pivot(index="timestamp", columns="station", values=feature)
        .reindex(columns=stations).to_numpy(np.float32)
        for feature in ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
    }
    for pollutant in ("PM10", "SO2", "NO2", "CO", "O3"):
        output.extend(_coupling_metrics(
            "UCI-Beijing", uci_panels["PM2.5"], uci_panels[pollutant], pollutant, 1
        ))

    # KDD Beijing distributes an aligned hourly series for all six pollutants.
    records, in_data = [], False
    path = root / "data/benchmarks/beijing_kdd/kdd_cup_2018_dataset_with_missing_values.tsf"
    for raw in path.open():
        line = raw.strip()
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or not line or line.startswith(("@", "#")):
            continue
        _, city, station, pollutant, start, raw_values = line.split(":", 5)
        if city != "Beijing":
            continue
        values = np.asarray(
            [np.nan if value == "?" else float(value) for value in raw_values.split(",")],
            dtype=np.float32,
        )
        records.append((station, pollutant, pd.to_datetime(start), values))
    kdd_panels = {}
    for pollutant in ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3"):
        series = [
            pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), name=station)
            for station, name, start, values in records if name == pollutant
        ]
        kdd_panels[pollutant] = pd.concat(series, axis=1).sort_index()
    common_stations = sorted(set.intersection(*[set(frame.columns) for frame in kdd_panels.values()]))
    for pollutant in ("PM10", "SO2", "NO2", "CO", "O3"):
        output.extend(_coupling_metrics(
            "KDD-Beijing",
            kdd_panels["PM2.5"][common_stations].to_numpy(np.float32),
            kdd_panels[pollutant][common_stations].to_numpy(np.float32),
            pollutant, 1,
        ))

    base = root / "data/benchmarks/airqualitybench"
    with h5py.File(base / "aq_compact_2025.h5") as handle:
        values = handle["values"][:, aq_selected, :].astype(np.float32)
        masks = handle["masks"][:, aq_selected, :].astype(bool)
        names = [value.decode() for value in handle["params"][:]]
    values[~masks] = np.nan
    pm = values[..., names.index("pm25")]
    for pollutant in ("pm10", "so2", "no2", "co", "o3"):
        output.extend(_coupling_metrics(
            "AirQualityBench", pm, values[..., names.index(pollutant)], pollutant.upper(), 1
        ))
    return output


def _meteorology_precursors(root: Path) -> list[dict]:
    output = []
    frame = load_raw_frames(root / "data/raw/PRSA_Data_20130301-20170228").sort_values(
        ["station", "timestamp"]
    )
    angle = frame["wd"].map({
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
        "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }) * np.pi / 180
    frame["wind_east"] = -frame.WSPM * np.sin(angle)
    frame["wind_north"] = -frame.WSPM * np.cos(angle)
    stations = sorted(frame.station.unique())
    uci_features = {
        "temperature": "TEMP", "pressure": "PRES", "dewpoint": "DEWP",
        "precipitation": "RAIN", "wind_speed": "WSPM",
        "wind_east": "wind_east", "wind_north": "wind_north",
    }
    pm = frame.pivot(index="timestamp", columns="station", values="PM2.5").reindex(columns=stations).to_numpy(np.float32)
    for generic, feature in uci_features.items():
        values = frame.pivot(index="timestamp", columns="station", values=feature).reindex(columns=stations).to_numpy(np.float32)
        for horizon in HORIZON_HOURS:
            step = horizon
            delta = pm[step:] - pm[:-step]
            raw, partial = [], []
            for node in range(pm.shape[1]):
                raw.append(_rho(values[:-step, node], delta[:, node]))
                partial.append(_partial_rank_rho(values[:-step, node], delta[:, node], pm[:-step, node]))
            output.append({
                "dataset": "UCI-Beijing", "feature": generic, "horizon_h": horizon,
                "rho_median_node": float(np.nanmedian(raw)),
                "partial_current_pm_rho_median_node": float(np.nanmedian(partial)),
            })

    array = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    pm = np.asarray(array[:, :, -1], dtype=np.float32)
    knowair_features = {
        "temperature": np.asarray(array[:, :, 3], dtype=np.float32),
        "pressure": np.asarray(array[:, :, 9], dtype=np.float32),
        "dewpoint": np.asarray(array[:, :, 2], dtype=np.float32),
        "precipitation": np.asarray(array[:, :, 12], dtype=np.float32),
        "wind_speed": np.hypot(array[:, :, 13], array[:, :, 14]).astype(np.float32),
        "wind_east": np.asarray(array[:, :, 13], dtype=np.float32),
        "wind_north": np.asarray(array[:, :, 14], dtype=np.float32),
        "boundary_layer_height": np.asarray(array[:, :, 4], dtype=np.float32),
    }
    for feature, values in knowair_features.items():
        for horizon in HORIZON_HOURS:
            step = horizon // 3
            delta = pm[step:] - pm[:-step]
            raw, partial = [], []
            for node in range(pm.shape[1]):
                raw.append(_rho(values[:-step, node], delta[:, node]))
                partial.append(_partial_rank_rho(values[:-step, node], delta[:, node], pm[:-step, node]))
            output.append({
                "dataset": "KnowAir", "feature": feature, "horizon_h": horizon,
                "rho_median_node": float(np.nanmedian(raw)),
                "partial_current_pm_rho_median_node": float(np.nanmedian(partial)),
            })
    return output


def _airformer_diagnostics(root: Path):
    base = root / "data/benchmarks/airformer/extracted/AIR_TINY"
    arrays = {split: np.load(base / f"{split}.npz") for split in ("train", "val", "test")}
    with open(root / "data/benchmarks/airformer/extracted/sensor_graph/adj_mx_air_tiny.pkl", "rb") as handle:
        adjacency = pickle.load(handle, encoding="latin1")[-1]
    neighbours = _top_weight_neighbours(adjacency, 10)
    train_x = arrays["train"]["x"][..., 0]
    scale = np.quantile(train_x, .75, axis=(0, 1)) - np.quantile(train_x, .25, axis=(0, 1))
    scale[scale < 1e-6] = np.nan
    x = arrays["test"]["x"][..., 0]
    y = arrays["test"]["y"][..., 0]
    current = x[:, -1, :]
    recent = x[:, -1, :] - x[:, -2, :]
    neighbour_current = np.nanmean(current[:, neighbours], axis=2)
    neighbour_recent = np.nanmean(recent[:, neighbours], axis=2)
    horizon_rows = []
    for horizon in (1, 6, 24):
        future = y[:, horizon - 1, :]
        delta = future - current
        regional = delta.mean(axis=1)
        local = delta - regional[:, None]
        regional_var, local_var = np.var(regional), np.var(local)
        horizon_rows.append({
            "dataset": "AirFormer-tiny",
            "horizon_h": horizon,
            "persistence_rho": _rho(current, future),
            "persistence_mae_over_train_iqr": float(np.nanmedian(np.abs(delta) / scale)),
            "current_level_vs_future_change_rho": _rho(current, delta),
            "own_recent_trend_rho_median_node": float(np.nanmedian([
                _rho(recent[:, node], delta[:, node]) for node in range(current.shape[1])
            ])),
            "neighbour_recent_trend_rho_median_node": float(np.nanmedian([
                _rho(neighbour_recent[:, node], delta[:, node]) for node in range(current.shape[1])
            ])),
            "spatial_gap_rho_median_node": float(np.nanmedian([
                _rho(neighbour_current[:, node] - current[:, node], delta[:, node])
                for node in range(current.shape[1])
            ])),
            "spatial_gap_partial_current_rho_median_node": float(np.nanmedian([
                _partial_rank_rho(
                    neighbour_current[:, node] - current[:, node],
                    delta[:, node], current[:, node],
                )
                for node in range(current.shape[1])
            ])),
            "cross_node_mean_component_share": float(regional_var / (regional_var + local_var)),
        })
    train_y = arrays["train"]["y"][..., 0]
    test_y = arrays["test"]["y"][..., 0]
    train_values = np.concatenate([train_x.reshape(-1), train_y.reshape(-1)])
    overview = {
        "dataset": "AirFormer-tiny", "scope": "nationwide packaged windows",
        "time_steps": 60 * 48, "nodes": x.shape[-1], "cadence_h": 1,
        "start": None, "end": None, "coverage_mean": 1.0,
        "coverage_node_p10": 1.0, "coverage_node_p90": 1.0,
        "pm_p50": float(np.median(train_values)),
        "pm_p90": float(np.quantile(train_values, .90)),
        "pm_p99": float(np.quantile(train_values, .99)),
        "tail_p99_over_p90": float(np.quantile(train_values, .99) / np.quantile(train_values, .90)),
    }
    limitations = [{
        "dataset": "AirFormer-tiny", "event_rate_test": np.nan,
        "local_event_coherence": np.nan,
        "local_event_expected_if_time_independent": np.nan,
        "local_event_lift_over_independent": np.nan,
        "active_node_fraction_median": np.nan, "active_node_fraction_p90": np.nan,
        "spike_duration_h_median": np.nan, "spike_duration_h_p90": np.nan,
        "coverage_mean": 1.0, "missing_gap_h_p90": 0, "missing_gap_h_max": 0,
        "timestamps_10pct_nodes_missing_rate": 0,
        "seasonal_amplitude_over_iqr": np.nan,
        "diurnal_amplitude_over_iqr": np.nan,
        "weekend_minus_weekday_over_iqr": np.nan,
        "node_median_spread_over_global_iqr": float(
            np.std(np.median(train_x, axis=(0, 1)))
            / (np.quantile(train_x, .75) - np.quantile(train_x, .25))
        ),
        "native_change_up_p99_over_down_p99": np.nan,
        "test_minus_train_median_over_train_iqr": float(
            (np.median(test_y) - np.median(train_y))
            / (np.quantile(train_y, .75) - np.quantile(train_y, .25))
        ),
        "limitation": "Only 20 windows per split; event, season and episode statistics are invalid.",
    }]
    return overview, horizon_rows, limitations, []


def _make_cross_figure(output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    horizon = pd.read_csv(output_dir / "horizon_signals.csv")
    edge = pd.read_csv(output_dir / "edge_lead_lag.csv")
    events = pd.read_csv(output_dir / "events_missingness_shift.csv")
    onset = events.dropna(subset=["current_over_p90_bin"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.lineplot(
        data=horizon, x="horizon_h", y="persistence_rho", hue="dataset",
        marker="o", ax=axes[0, 0],
    )
    axes[0, 0].set_title("Persistence decays with forecast horizon")
    sns.lineplot(
        data=horizon.dropna(subset=["spatial_gap_partial_current_rho_median_node"]),
        x="horizon_h", y="spatial_gap_partial_current_rho_median_node",
        hue="dataset", marker="o", ax=axes[0, 1],
    )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Spatial gap association after controlling current PM")
    sns.lineplot(
        data=edge, x="lag_h", y="median_rho", hue="dataset", marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Neighbour change relation is predominantly synchronous")
    sns.lineplot(
        data=onset, x="current_over_p90_bin", y="onset_24h_rate",
        hue="dataset", marker="o", ax=axes[1, 1],
    )
    axes[1, 1].set_title("24-hour onset probability vs current station-relative level")
    axes[1, 1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(output_dir / "cross_dataset_main.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_cross_dataset_eda(
    root: str | Path = ".", output_dir: str | Path = "artifacts/cross_dataset_eda"
) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overviews, horizons, events, edges = [], [], [], []
    robustness, distance_pairs, distance_summary = [], [], []

    uci = _load_uci(root)
    analyses = [("UCI-Beijing", *uci, 1, "Beijing city, 12 stations")]
    # Reorder tuple to the common panel/timestamps/neighbours/cadence/scope form.
    analyses = [(name, panel, cadence, neighbours, timestamps, scope, coordinates)
                for name, panel, timestamps, neighbours, coordinates, cadence, scope in analyses]

    kdd = _parse_kdd(root / "data/benchmarks/beijing_kdd/kdd_cup_2018_dataset_with_missing_values.tsf")
    coords = pd.read_csv(root / "data/benchmarks/beijing_kdd/beijing_station_coords.csv").set_index("station")
    bj = kdd["Beijing"]
    bj_coordinates = coords.loc[bj.columns, ["longitude", "latitude"]].to_numpy()
    analyses.append(("KDD-Beijing", bj.to_numpy(np.float32), 1,
                     _knn(coords.loc[bj.columns, ["longitude", "latitude"]].to_numpy(), 6),
                     bj.index, "Beijing city, 35 stations", bj_coordinates))
    london = kdd["London"]
    # No station coordinates are distributed in the TSF package.  The other-city
    # context is represented by a deterministic all-node subset.
    london_neighbours = np.stack([
        np.asarray([j for j in range(london.shape[1]) if j != i])
        for i in range(london.shape[1])
    ])
    analyses.append(("KDD-London", london.to_numpy(np.float32), 1,
                     london_neighbours, london.index, "London city, 24 PM2.5 stations", None))

    knowair = _load_knowair(root)
    analyses.append(("KnowAir", knowair[0], 3, knowair[2], knowair[1],
                     "China, 184 cities", knowair[3]))
    aq = _load_airqualitybench(root)
    analyses.append(("AirQualityBench", aq[0], 1, aq[2], aq[1],
                     "global, 250 highest-coverage PM2.5 stations", aq[3]))

    for name, panel, cadence, neighbours, timestamps, scope, coordinates in analyses:
        result = _panel_diagnostics(name, panel, cadence, neighbours, timestamps, scope)
        overview, horizon, event, edge = result
        overviews.append(overview)
        horizons.extend(horizon)
        events.extend(event)
        edges.extend(edge)
        robustness.extend(
            _horizon_regime_diagnostics(
                name, panel, cadence, neighbours, timestamps
            )
        )
        if coordinates is not None:
            pair_rows, pair_summary = _distance_diagnostics(
                name, panel, coordinates
            )
            distance_pairs.extend(pair_rows)
            distance_summary.append(pair_summary)

    airformer = _airformer_diagnostics(root)
    overviews.append(airformer[0])
    horizons.extend(airformer[1])
    events.extend(airformer[2])
    edges.extend(airformer[3])

    overview_frame = pd.DataFrame(overviews)
    horizon_frame = pd.DataFrame(horizons)
    event_frame = pd.DataFrame(events)
    edge_frame = pd.DataFrame(edges)
    robustness_frame = pd.DataFrame(robustness)
    distance_pair_frame = pd.DataFrame(distance_pairs)
    distance_summary_frame = pd.DataFrame(distance_summary)
    overview_frame.to_csv(output_dir / "dataset_overview.csv", index=False)
    horizon_frame.to_csv(output_dir / "horizon_signals.csv", index=False)
    event_frame.to_csv(output_dir / "events_missingness_shift.csv", index=False)
    edge_frame.to_csv(output_dir / "edge_lead_lag.csv", index=False)
    robustness_frame.to_csv(output_dir / "horizon_regime_robustness.csv", index=False)
    distance_pair_frame.to_csv(output_dir / "distance_pair_correlations.csv", index=False)
    distance_summary_frame.to_csv(output_dir / "distance_decay_summary.csv", index=False)

    knowair_wind_pairs, knowair_wind_robust = _knowair_wind_alignment(
        root, knowair[2], knowair[3]
    )
    wind_pairs = pd.DataFrame(knowair_wind_pairs)
    wind_summary = wind_pairs.groupby(["dataset", "lag_h"]).aligned_minus_opposed.agg(
        pair_count="count", mean="mean", median="median", std="std",
        positive_pair_fraction=lambda values: (values > 0).mean(),
    ).reset_index()
    uci_wind_path = root / "artifacts/deep_eda_uci/wind_alignment_contrast.csv"
    if uci_wind_path.exists():
        uci_wind = pd.read_csv(uci_wind_path).rename(columns={"count": "pair_count"})
        uci_wind.insert(0, "dataset", "UCI-Beijing")
        wind_summary = pd.concat([uci_wind[wind_summary.columns], wind_summary], ignore_index=True)
    wind_pairs.to_csv(output_dir / "knowair_wind_alignment_pairs.csv", index=False)
    wind_summary.to_csv(output_dir / "wind_alignment_replication.csv", index=False)
    wind_robust_frame = pd.DataFrame(knowair_wind_robust)
    wind_robust_summary = wind_robust_frame.groupby(["dataset", "stratum"]).aligned_minus_opposed.agg(
        pair_count="count", mean="mean", median="median", std="std",
        positive_pair_fraction=lambda values: (values > 0).mean(),
    ).reset_index()
    wind_robust_summary.to_csv(output_dir / "knowair_wind_robustness.csv", index=False)

    pollutant_coupling = pd.DataFrame(_pollutant_coupling(root, aq[4]))
    pollutant_coupling.to_csv(output_dir / "pollutant_coupling.csv", index=False)
    meteorology = pd.DataFrame(_meteorology_precursors(root))
    meteorology.to_csv(output_dir / "meteorology_precursors.csv", index=False)
    _make_cross_figure(output_dir)

    summary = {
        "scope": "Five benchmark families; KDD cities are reported separately.",
        "caveats": [
            "All results are descriptive and hypothesis-generating.",
            "AirFormer tiny is not a population sample and cannot support event or seasonal claims.",
            "AirQualityBench diagnostics use 250 highest-coverage PM2.5 stations.",
            "KDD London lacks coordinates in the distributed TSF; its context is not geographic KNN.",
            "Cross-node mean component share has different spatial scope and must not be compared as a universal physical parameter.",
        ],
        "datasets": overview_frame.to_dict(orient="records"),
    }
    (output_dir / "cross_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary
