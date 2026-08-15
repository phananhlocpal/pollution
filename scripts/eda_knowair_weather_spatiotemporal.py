"""Final deep EDA of 72h past/future KnowAir weather, within and across stations.

This is deliberately split-agnostic and uses every overlapping 72h -> 72h
window. Dynamic analyses operate on station x month x dataset-clock-hour
anomalies. Spatial lead-lag diagnostics distinguish synchronous coherence from
incremental source-station information after controlling target weather now.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


RAW = (
    "u100", "v100", "dewpoint", "temperature", "pbl", "k_index", "rh950",
    "rh975", "specific_humidity950", "pressure", "temperature925",
    "temperature950", "precipitation", "u950", "v950", "vertical_velocity950",
    "vorticity950", "pm25",
)
LABELS = {
    "temperature": "Temperature", "pressure": "Pressure", "rh950": "RH 950 hPa",
    "wind_speed": "Wind speed", "wind_u": "Wind u", "wind_v": "Wind v",
    "pbl": "Boundary-layer height", "ventilation": "Ventilation",
    "dewpoint_deficit": "Dewpoint deficit", "log_precipitation": "Log precipitation",
}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SPATIAL_FEATURES = ("temperature", "pressure", "rh950", "wind_speed", "wind_u", "wind_v", "pbl")
ANALOG_FEATURES = ("temperature", "pressure", "rh950", "wind_u", "wind_v", "pbl", "log_precipitation")


def station_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(0, keepdims=True)
    yc = y - y.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(xc).sum(0) * np.square(yc).sum(0))
    return np.divide((xc * yc).sum(0), denominator, out=np.full(x.shape[1], np.nan),
                     where=denominator > 1e-12)


def station_partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> np.ndarray:
    rxy, rxz, ryz = station_corr(x, y), station_corr(x, control), station_corr(y, control)
    denominator = np.sqrt(np.maximum((1 - rxz * rxz) * (1 - ryz * ryz), 0.0))
    return np.divide(rxy - rxz * ryz, denominator, out=np.full_like(rxy, np.nan),
                     where=denominator > 1e-12)


def summary(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = values[np.isfinite(values)]
    return {f"{prefix}median": float(np.median(values)),
            f"{prefix}q25": float(np.quantile(values, .25)),
            f"{prefix}q75": float(np.quantile(values, .75)),
            f"{prefix}stations": int(len(values))}


def climatology_anomaly(values: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    groups = timestamps.month.to_numpy() * 100 + timestamps.hour.to_numpy()
    for group in np.unique(groups):
        select = groups == group
        result[select] = values[select] - values[select].mean(0, keepdims=True)
    return result


def window_mean(values: np.ndarray, start: int, length: int, windows: int) -> np.ndarray:
    cumulative = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0)
    return (cumulative[start + length:start + length + windows]
            - cumulative[start:start + windows]) / length


def cross_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = x - x.mean(0), y - y.mean(0)
    norm = np.sqrt(np.square(xc).sum(0)[:, None] * np.square(yc).sum(0)[None, :])
    return np.divide(xc.T @ yc, norm, out=np.full((x.shape[1], y.shape[1]), np.nan),
                     where=norm > 1e-12)


def spatial_geometry(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlon = lon[None, :] - lon[:, None]
    dlat = lat[None, :] - lat[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    east = dlon * np.cos((lat[:, None] + lat[None, :]) / 2)
    bearing = np.degrees(np.arctan2(east, dlat)) % 360
    return distance, bearing


def bearing_sector(bearing: np.ndarray) -> np.ndarray:
    names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return names[((bearing + 22.5) // 45).astype(int) % 8]


def station_quintiles(values: np.ndarray) -> np.ndarray:
    labels = np.empty(values.shape, dtype=np.int8)
    for station in range(values.shape[1]):
        edges = np.quantile(values[:, station], [.2, .4, .6, .8])
        labels[:, station] = np.searchsorted(edges, values[:, station], side="right")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/knowair_weather_spatiotemporal_eda")
    parser.add_argument("--analog-pairs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.root) / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    raw = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    timestamps = pd.date_range("2015-01-01", periods=len(raw), freq="3h")
    coords = np.loadtxt(root / "data/benchmarks/knowair/city.txt", usecols=(2, 3))
    stations = np.loadtxt(root / "data/benchmarks/knowair/city.txt", usecols=(1,), dtype=str)
    get = lambda name: np.asarray(raw[..., RAW.index(name)], dtype=np.float64)
    u, v = get("u100"), get("v100")
    physical = {
        "temperature": get("temperature") - 273.15,
        "pressure": get("pressure") / 100,
        "rh950": get("rh950"), "wind_speed": np.hypot(u, v),
        "wind_u": u, "wind_v": v, "pbl": get("pbl"),
        "ventilation": np.hypot(u, v) * get("pbl"),
        "dewpoint_deficit": get("temperature") - get("dewpoint"),
        "log_precipitation": np.log1p(np.maximum(get("precipitation") * 1000, 0)),
    }
    anomaly = {name: climatology_anomaly(values, timestamps) for name, values in physical.items()}
    standardized = {}
    for name, values in anomaly.items():
        scale = values.std(0, keepdims=True)
        standardized[name] = values / np.maximum(scale, 1e-12)

    windows = len(raw) - 47
    origin_times = timestamps[23:23 + windows]
    past_hours = np.arange(-69, 1, 3)
    lead_hours = np.arange(3, 73, 3)

    # ------------------------------------------------------------------
    # Within-station memory, heterogeneity, seasonal stability.
    persistence_rows, station_rows, seasonal_rows = [], [], []
    for feature, values in anomaly.items():
        lead_correlations = []
        for lead_step, lead_hour in enumerate(lead_hours, start=1):
            r = station_corr(values[:-lead_step], values[lead_step:])
            lead_correlations.append(r)
            stats = summary(r)
            persistence_rows.append({"feature": feature, "lead_hour": int(lead_hour), **stats})
            for station, value in enumerate(r):
                station_rows.append({"feature": feature, "station_index": station,
                                     "station": stations[station], "longitude": coords[station, 0],
                                     "latitude": coords[station, 1], "lead_hour": int(lead_hour), "r": value})
        lead_correlations = np.stack(lead_correlations)
        for station in range(len(stations)):
            below = np.flatnonzero(lead_correlations[:, station] < .5)
            half_life = int(lead_hours[below[0]]) if len(below) else 75
            station_rows.append({"feature": feature, "station_index": station,
                                 "station": stations[station], "longitude": coords[station, 0],
                                 "latitude": coords[station, 1], "lead_hour": 0,
                                 "r": np.nan, "first_lead_below_r_0.5": half_life})
        for season in ("DJF", "MAM", "JJA", "SON"):
            mask = np.array([SEASONS[m] == season for m in timestamps.month])
            for lead_step in (1, 8, 16, 24):
                valid = mask[:-lead_step] & mask[lead_step:]
                r = station_corr(values[:-lead_step][valid], values[lead_step:][valid])
                seasonal_rows.append({"feature": feature, "season": season,
                                      "lead_hour": lead_step * 3, **summary(r)})
    persistence = pd.DataFrame(persistence_rows)
    station_persistence = pd.DataFrame(station_rows)
    seasonal = pd.DataFrame(seasonal_rows)
    persistence.to_csv(output / "within_station_persistence.csv", index=False)
    station_persistence.to_csv(output / "station_persistence_detail.csv", index=False)
    seasonal.to_csv(output / "seasonal_weather_persistence.csv", index=False)
    geographic_rows = []
    for feature in physical:
        for lead_hour in (3, 24, 48, 72):
            subset = station_persistence[(station_persistence.feature == feature)
                                         & (station_persistence.lead_hour == lead_hour)]
            geographic_rows.append({
                "feature": feature, "lead_hour": lead_hour,
                "spearman_r_with_longitude": float(spearmanr(subset.longitude, subset.r).statistic),
                "spearman_r_with_latitude": float(spearmanr(subset.latitude, subset.r).statistic),
                "min_station_r": float(subset.r.min()), "max_station_r": float(subset.r.max()),
            })
    geographic = pd.DataFrame(geographic_rows)
    geographic.to_csv(output / "geographic_persistence_gradients.csv", index=False)

    # Which exact part of the 72h history predicts each future day mean?
    relevance_rows = []
    for feature, values in anomaly.items():
        future_days = [window_mean(values, 24 + day * 8, 8, windows) for day in range(3)]
        for past_index, past_hour in enumerate(past_hours):
            past_value = values[past_index:past_index + windows]
            for day, future_day in enumerate(future_days, start=1):
                r = station_corr(past_value, future_day)
                relevance_rows.append({"feature": feature, "past_hour": int(past_hour),
                                       "future_day": day, **summary(r)})
    relevance = pd.DataFrame(relevance_rows)
    relevance.to_csv(output / "history_offset_relevance.csv", index=False)

    # Trend information after controlling recent state.
    trend_rows = []
    for feature, values in anomaly.items():
        previous = window_mean(values, 8, 8, windows)
        recent = window_mean(values, 16, 8, windows)
        trend = recent - previous
        for day in range(3):
            future = window_mean(values, 24 + day * 8, 8, windows)
            raw_r = station_corr(trend, future)
            partial = station_partial_corr(trend, future, recent)
            trend_rows.append({"feature": feature, "future_day": day + 1,
                               **summary(raw_r, "raw_"), **summary(partial, "partial_")})
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(output / "weather_trend_value_add.csv", index=False)

    # Nonlinear station-specific regime transition matrices.
    transition_rows = []
    for feature, values in anomaly.items():
        recent = window_mean(values, 16, 8, windows)
        past_bins = station_quintiles(recent)
        for day in (1, 3):
            future = window_mean(values, 24 + (day - 1) * 8, 8, windows)
            future_bins = station_quintiles(future)
            for source in range(5):
                station_probabilities = []
                for destination in range(5):
                    probabilities = []
                    for station in range(len(stations)):
                        select = past_bins[:, station] == source
                        probabilities.append(np.mean(future_bins[select, station] == destination))
                    probabilities = np.asarray(probabilities)
                    transition_rows.append({"feature": feature, "future_day": day,
                                            "past_quintile": source + 1,
                                            "future_quintile": destination + 1,
                                            **summary(probabilities)})
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(output / "station_regime_transitions.csv", index=False)

    # Multivariate analog recurrence: do similar non-overlapping histories lead
    # to similar futures at the same station? Pair sampling is shared across sites.
    rng = np.random.default_rng(args.seed)
    left, right = [], []
    while len(left) < args.analog_pairs:
        a = int(rng.integers(0, windows))
        b = int(rng.integers(0, windows))
        if abs(a - b) >= 48:  # no shared 72h past/future support
            left.append(a); right.append(b)
    left, right = np.asarray(left), np.asarray(right)
    analog_rows = []
    for station in range(len(stations)):
        past_full, past_recent, future_day1, future_day3, future_full = [], [], [], [], []
        for feature in ANALOG_FEATURES:
            x = standardized[feature][:, station]
            def pair_rmse(starts: np.ndarray, offset: int, length: int) -> np.ndarray:
                difference = np.stack([x[starts + offset + k] - x[right + offset + k]
                                       for k in range(length)], axis=1)
                return np.square(difference).mean(1)
            past_full.append(pair_rmse(left, 0, 24))
            past_recent.append(pair_rmse(left, 16, 8))
            future_day1.append(pair_rmse(left, 24, 8))
            future_day3.append(pair_rmse(left, 40, 8))
            future_full.append(pair_rmse(left, 24, 24))
        distances = {"past_72h": np.sqrt(np.mean(past_full, axis=0)),
                     "past_recent_24h": np.sqrt(np.mean(past_recent, axis=0)),
                     "future_day1": np.sqrt(np.mean(future_day1, axis=0)),
                     "future_day3": np.sqrt(np.mean(future_day3, axis=0)),
                     "future_72h": np.sqrt(np.mean(future_full, axis=0))}
        for history in ("past_72h", "past_recent_24h"):
            for future in ("future_day1", "future_day3", "future_72h"):
                rho = spearmanr(distances[history], distances[future]).statistic
                analog_rows.append({"station_index": station, "station": stations[station],
                                    "longitude": coords[station, 0], "latitude": coords[station, 1],
                                    "history_distance": history, "future_distance": future,
                                    "spearman_distance_rho": float(rho)})
    analog = pd.DataFrame(analog_rows)
    analog.to_csv(output / "multivariate_analog_predictability.csv", index=False)

    # ------------------------------------------------------------------
    # Regional/common modes versus station-local residuals.
    common_rows, pca_rows, pc_memory_rows = [], [], []
    selected_leads = (1, 8, 16, 24)
    for feature in SPATIAL_FEATURES:
        x = standardized[feature]
        covariance = x.T @ x / (len(x) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        explained = eigenvalues / eigenvalues.sum()
        for component in range(10):
            pca_rows.append({"feature": feature, "component": component + 1,
                             "explained_variance_ratio": float(explained[component]),
                             "cumulative_explained_variance": float(explained[:component + 1].sum())})
        scores = x @ eigenvectors[:, :10]
        regional = scores[:, :3] @ eigenvectors[:, :3].T
        local = x - regional
        for lead_step in selected_leads:
            for representation, values in (("total", x), ("regional_top3", regional),
                                           ("local_after_top3", local)):
                r = station_corr(values[:-lead_step], values[lead_step:])
                common_rows.append({"feature": feature, "representation": representation,
                                    "lead_hour": lead_step * 3, **summary(r)})
            for component in range(3):
                pc = scores[:, component]
                pc_memory_rows.append({"feature": feature, "component": component + 1,
                                       "lead_hour": lead_step * 3,
                                       "r": float(np.corrcoef(pc[:-lead_step], pc[lead_step:])[0, 1])})
    common = pd.DataFrame(common_rows)
    pca = pd.DataFrame(pca_rows)
    pc_memory = pd.DataFrame(pc_memory_rows)
    common.to_csv(output / "regional_vs_local_memory.csv", index=False)
    pca.to_csv(output / "spatial_pca_explained_variance.csv", index=False)
    pc_memory.to_csv(output / "regional_pc_temporal_memory.csv", index=False)

    # ------------------------------------------------------------------
    # Cross-station spatial coherence and directed incremental lead-lag graph.
    distance, bearing = spatial_geometry(coords)
    sectors = bearing_sector(bearing)
    distance_edges = np.array([0, 50, 100, 200, 400, 800, 1600, np.inf])
    distance_names = np.array(["<50", "50-100", "100-200", "200-400", "400-800", "800-1600", ">=1600"])
    distance_bin = distance_names[np.clip(np.digitize(distance, distance_edges[1:-1]), 0, len(distance_names) - 1)]
    off_diagonal = ~np.eye(len(stations), dtype=bool)
    spatial_rows, best_lag_rows = [], []
    directed_leads = (1, 2, 4, 8)
    for feature in SPATIAL_FEATURES:
        x = anomaly[feature]
        corr0 = cross_corr_matrix(x, x)
        lag_corrs, partials = {}, {}
        for lead_step in directed_leads:
            cross = cross_corr_matrix(x[:-lead_step], x[lead_step:])
            self_memory = np.diag(cross)[None, :]
            denominator = np.sqrt(np.maximum((1 - corr0 * corr0) * (1 - self_memory * self_memory), 0))
            partial = np.divide(cross - corr0 * self_memory, denominator,
                                out=np.full_like(cross, np.nan), where=denominator > 1e-12)
            lag_corrs[lead_step], partials[lead_step] = cross, partial
            for dist_name in distance_names:
                for sector in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
                    select = off_diagonal & (distance_bin == dist_name) & (sectors == sector)
                    if select.sum() == 0:
                        continue
                    spatial_rows.append({"feature": feature, "lead_hour": lead_step * 3,
                                         "distance_bin": dist_name, "bearing_source_to_target": sector,
                                         "edges": int(select.sum()),
                                         "median_cross_r": float(np.nanmedian(cross[select])),
                                         "median_partial_r": float(np.nanmedian(partial[select])),
                                         "q25_partial_r": float(np.nanquantile(partial[select], .25)),
                                         "q75_partial_r": float(np.nanquantile(partial[select], .75))})
        stack = np.stack([partials[lead] for lead in directed_leads])
        safe_stack = np.where(np.isfinite(stack), stack, -np.inf)
        best_index = np.argmax(safe_stack, axis=0)
        best_value = np.take_along_axis(safe_stack, best_index[None], axis=0)[0]
        for source, target in zip(*np.where(off_diagonal)):
            best_lag_rows.append({"feature": feature, "source_index": source,
                                  "source": stations[source], "target_index": target,
                                  "target": stations[target], "distance_km": distance[source, target],
                                  "bearing_source_to_target": sectors[source, target],
                                  "best_lead_hour": directed_leads[best_index[source, target]] * 3,
                                  "best_partial_r": best_value[source, target],
                                  "synchronous_r": corr0[source, target]})
        # Synchronous spatial range, without direction duplication.
        upper = np.triu(np.ones_like(corr0, dtype=bool), 1)
        for dist_name in distance_names:
            select = upper & (distance_bin == dist_name)
            spatial_rows.append({"feature": feature, "lead_hour": 0,
                                 "distance_bin": dist_name, "bearing_source_to_target": "all",
                                 "edges": int(select.sum()),
                                 "median_cross_r": float(np.nanmedian(corr0[select])),
                                 "median_partial_r": np.nan,
                                 "q25_partial_r": np.nan, "q75_partial_r": np.nan})
    spatial = pd.DataFrame(spatial_rows)
    best_lags = pd.DataFrame(best_lag_rows)
    spatial.to_csv(output / "cross_station_spatial_summary.csv", index=False)
    best_lags.to_csv(output / "directed_best_lag_edges.csv", index=False)

    # Directional asymmetry: compare source->target with reverse target->source.
    asymmetry_rows = []
    for feature in SPATIAL_FEATURES:
        subset = best_lags[best_lags.feature == feature]
        matrix = np.full((len(stations), len(stations)), np.nan)
        matrix[subset.source_index, subset.target_index] = subset.best_partial_r
        asymmetry = matrix - matrix.T
        for sector in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
            select = off_diagonal & (sectors == sector)
            asymmetry_rows.append({"feature": feature, "bearing_source_to_target": sector,
                                   "edges": int(select.sum()),
                                   "median_best_partial_r": float(np.nanmedian(matrix[select])),
                                   "median_directional_advantage_over_reverse": float(np.nanmedian(asymmetry[select]))})
    asymmetry = pd.DataFrame(asymmetry_rows)
    asymmetry.to_csv(output / "directional_asymmetry.csv", index=False)

    # Fixed-lag, year-wise check of the strongest apparent directional pattern.
    # This prevents a single year's synoptic regime from driving the conclusion.
    yearly_direction_rows = []
    spatial_band = off_diagonal & (distance >= 100) & (distance < 800)
    for feature in ("pressure", "temperature", "rh950", "wind_speed"):
        values = anomaly[feature]
        for year in range(2015, 2019):
            time_mask = timestamps.year == year
            lead_step = 2
            valid = time_mask[:-lead_step] & time_mask[lead_step:]
            now, future = values[:-lead_step][valid], values[lead_step:][valid]
            corr0_year = cross_corr_matrix(now, now)
            cross_year = cross_corr_matrix(now, future)
            self_memory = np.diag(cross_year)[None, :]
            denominator = np.sqrt(np.maximum((1 - corr0_year * corr0_year)
                                             * (1 - self_memory * self_memory), 0))
            partial_year = np.divide(cross_year - corr0_year * self_memory, denominator,
                                     out=np.full_like(cross_year, np.nan), where=denominator > 1e-12)
            for sector in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
                select = spatial_band & (sectors == sector)
                yearly_direction_rows.append({"feature": feature, "year": year, "lead_hour": 6,
                                              "distance_range_km": "100-800",
                                              "bearing_source_to_target": sector,
                                              "edges": int(select.sum()),
                                              "median_partial_r": float(np.nanmedian(partial_year[select]))})
    yearly_direction = pd.DataFrame(yearly_direction_rows)
    yearly_direction.to_csv(output / "yearly_directional_robustness.csv", index=False)

    # Figures.
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for feature in ("temperature", "pressure", "rh950", "wind_speed"):
        line = persistence[persistence.feature == feature]
        axes[0, 0].plot(line.lead_hour, line["median"], marker="o", ms=3, label=LABELS[feature])
    axes[0, 0].set(title="Within-station dynamic memory", xlabel="Lead (hours)", ylabel="Median station r")
    axes[0, 0].legend(fontsize=8)
    for day in (1, 2, 3):
        line = relevance[(relevance.feature == "temperature") & (relevance.future_day == day)]
        axes[0, 1].plot(line.past_hour, line["median"], marker="o", ms=3, label=f"future day {day}")
    axes[0, 1].set(title="Temperature: which past offset matters?", xlabel="Past hour", ylabel="Median station r")
    axes[0, 1].legend()
    analog_summary = (analog.groupby(["history_distance", "future_distance"])
                      .spearman_distance_rho.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
                      .reset_index())
    analog_summary.columns = ["history", "future", "median", "q25", "q75"]
    sns.barplot(data=analog_summary, x="future", y="median", hue="history", ax=axes[1, 0])
    axes[1, 0].set(title="Multivariate analog determinism", ylabel="Median station Spearman rho", xlabel="Future segment")
    plot_common = common[(common.feature == "temperature") & (common.representation.isin(["total", "local_after_top3"]))]
    sns.lineplot(data=plot_common, x="lead_hour", y="median", hue="representation", marker="o", ax=axes[1, 1])
    axes[1, 1].set(title="Temperature memory: regional versus local", ylabel="Median station r")
    fig.savefig(output / "within_station_hidden_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, ("temperature", "pressure", "rh950", "wind_speed")):
        same_time = spatial[(spatial.feature == feature) & (spatial.lead_hour == 0)]
        axis.plot(same_time.distance_bin, same_time.median_cross_r, marker="o")
        axis.set(title=LABELS[feature], xlabel="Station distance (km)", ylabel="Median synchronous r")
        axis.tick_params(axis="x", rotation=35)
    fig.suptitle("Spatial coherence length of weather anomalies", fontsize=16)
    fig.savefig(output / "spatial_coherence_by_distance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    direction_plot = spatial[(spatial.feature == "pressure") & (spatial.lead_hour == 6)
                             & (spatial.distance_bin.isin(["100-200", "200-400", "400-800"]))]
    sns.barplot(data=direction_plot, x="bearing_source_to_target", y="median_partial_r",
                hue="distance_bin", order=["N", "NE", "E", "SE", "S", "SW", "W", "NW"], ax=axes[0])
    axes[0].set(title="Pressure neighbour value-add at +6h", xlabel="Source -> target bearing", ylabel="Median partial r")
    lag_distribution = (best_lags.groupby(["feature", "best_lead_hour"]).size()
                        .rename("edges").reset_index())
    lag_distribution["fraction"] = (lag_distribution.edges
                                    / lag_distribution.groupby("feature").edges.transform("sum"))
    sns.barplot(data=lag_distribution, x="feature", y="fraction", hue="best_lead_hour", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(title="Best directed neighbour lag", ylabel="Fraction of directed edges")
    fig.savefig(output / "directed_cross_station_dynamics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, ("temperature", "pressure", "rh950", "wind_speed")):
        subset = pca[pca.feature == feature]
        axis.bar(subset.component, subset.explained_variance_ratio)
        axis.plot(subset.component, subset.cumulative_explained_variance, color="black", marker="o")
        axis.set(title=LABELS[feature], xlabel="Spatial PC", ylabel="Explained / cumulative fraction", ylim=(0, 1.05))
    fig.suptitle("Low-rank regional structure", fontsize=16)
    fig.savefig(output / "spatial_pca_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    station_24 = station_persistence[station_persistence.lead_hour == 24]
    for axis, feature in zip(axes.flat, ("temperature", "pressure", "rh950", "wind_speed")):
        subset = station_24[station_24.feature == feature]
        sns.regplot(data=subset, x="latitude", y="r", scatter_kws={"s": 22, "alpha": .7},
                    line_kws={"color": "black"}, ax=axis)
        rho = spearmanr(subset.latitude, subset.r).statistic
        axis.set(title=f"{LABELS[feature]} (Spearman rho={rho:.2f})",
                 ylabel="24h persistence r")
    fig.suptitle("Geographic heterogeneity of within-station memory", fontsize=16)
    fig.savefig(output / "station_persistence_geography.png", dpi=180)
    plt.close(fig)

    summary_payload = {
        "dataset": {"time_steps": len(raw), "stations": len(stations), "overlapping_windows": windows,
                    "split_boundaries_used": False, "anomaly": "station x month x dataset-clock-hour"},
        "analog_pairs_per_station": args.analog_pairs,
        "within_station_selected": persistence[persistence.lead_hour.isin([3, 24, 48, 72])].to_dict("records"),
        "trend_value_add": trends.to_dict("records"),
        "analog_summary": analog_summary.to_dict("records"),
        "pca_top3": pca[pca.component == 3][["feature", "cumulative_explained_variance"]].to_dict("records"),
        "directional_asymmetry": asymmetry.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(list(output.iterdir())),
                      "directed_edges": len(best_lags), "analog_rows": len(analog)}, indent=2))


if __name__ == "__main__":
    main()
