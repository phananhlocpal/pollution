"""Joint PM-weather EDA for all overlapping KnowAir 72h -> 72h windows.

The analysis separates within-station dynamics from cross-station transport.
PM <= 0 is missing. Dynamic variables are station/month/dataset-clock-hour
anomalies; PM uses log1p before anomaly removal.
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
WEATHER_CORE = ("temperature", "pressure", "rh950", "wind_speed", "wind_u", "wind_v",
                "pbl", "ventilation", "log_precipitation")
MODE_WEATHER = ("temperature", "pressure", "rh950", "wind_u", "wind_v", "pbl", "log_precipitation")
LABELS = {"pm": "PM2.5", "temperature": "Temperature", "pressure": "Pressure",
          "rh950": "RH 950 hPa", "wind_speed": "Wind speed", "wind_u": "Wind u",
          "wind_v": "Wind v", "pbl": "PBL height", "ventilation": "Ventilation",
          "log_precipitation": "Log precipitation"}


def climatology_anomaly(values: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    groups = timestamps.month.to_numpy() * 100 + timestamps.hour.to_numpy()
    for group in np.unique(groups):
        select = groups == group
        block = values[select]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=0, keepdims=True)
        output[select] = block - mean
    return output


def window_mean(values: np.ndarray, start: int, length: int, windows: int,
                minimum_fraction: float = .75) -> np.ndarray:
    valid = np.isfinite(values)
    total = np.concatenate([np.zeros((1, values.shape[1])),
                            np.cumsum(np.where(valid, values, 0.0), axis=0)], axis=0)
    count = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.int32),
                            np.cumsum(valid, axis=0, dtype=np.int32)], axis=0)
    sums = total[start + length:start + length + windows] - total[start:start + windows]
    counts = count[start + length:start + length + windows] - count[start:start + windows]
    return np.divide(sums, counts, out=np.full(sums.shape, np.nan),
                     where=counts >= int(np.ceil(length * minimum_fraction)))


def station_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(0)
    xv, yv = np.where(valid, x, 0), np.where(valid, y, 0)
    sx, sy = xv.sum(0), yv.sum(0)
    covariance = (xv * yv).sum(0) - sx * sy / np.maximum(count, 1)
    vx = np.square(xv).sum(0) - sx * sx / np.maximum(count, 1)
    vy = np.square(yv).sum(0) - sy * sy / np.maximum(count, 1)
    denominator = np.sqrt(np.maximum(vx * vy, 0))
    return np.divide(covariance, denominator, out=np.full(x.shape[1], np.nan),
                     where=(count >= 30) & (denominator > 1e-12))


def station_partial(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> np.ndarray:
    rxy, rxz, ryz = station_corr(x, y), station_corr(x, control), station_corr(y, control)
    denominator = np.sqrt(np.maximum((1 - rxz * rxz) * (1 - ryz * ryz), 0))
    return np.divide(rxy - rxz * ryz, denominator, out=np.full_like(rxy, np.nan),
                     where=denominator > 1e-12)


def stat(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}median": np.nan, f"{prefix}q25": np.nan,
                f"{prefix}q75": np.nan, f"{prefix}n": 0}
    return {f"{prefix}median": float(np.median(values)),
            f"{prefix}q25": float(np.quantile(values, .25)),
            f"{prefix}q75": float(np.quantile(values, .75)),
            f"{prefix}n": int(len(values))}


def adjusted_r2(y: np.ndarray, predictors: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(predictors).all(1)
    y, predictors = y[valid], predictors[valid]
    if len(y) < predictors.shape[1] + 30:
        return np.nan
    design = np.column_stack([np.ones(len(y)), predictors])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    total = np.square(y - y.mean()).sum()
    if total < 1e-12:
        return np.nan
    r2 = 1 - np.square(y - fitted).sum() / total
    n, p = len(y), predictors.shape[1]
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def multiple_partial(x: np.ndarray, y: np.ndarray, controls: np.ndarray,
                     select: np.ndarray | None = None) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(1)
    if select is not None:
        valid &= select
    if valid.sum() < controls.shape[1] + 50:
        return np.nan
    design = np.column_stack([np.ones(valid.sum()), controls[valid]])
    xr = x[valid] - design @ np.linalg.lstsq(design, x[valid], rcond=None)[0]
    yr = y[valid] - design @ np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return float(np.corrcoef(xr, yr)[0, 1])


def pca_scores(values: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.nanstd(values, axis=0, keepdims=True)
    x = np.nan_to_num(values / np.maximum(scale, 1e-12))
    covariance = x.T @ x / (len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors, eigenvalues = eigenvectors[:, order], eigenvalues[order]
    return x @ eigenvectors[:, :components], eigenvalues[:components] / eigenvalues.sum()


def spatial_geometry(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlon, dlat = lon[None, :] - lon[:, None], lat[None, :] - lat[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    east = dlon * np.cos((lat[:, None] + lat[None, :]) / 2)
    norm = np.hypot(east, dlat)
    return distance, np.divide(east, norm, out=np.zeros_like(east), where=norm > 0), \
        np.divide(dlat, norm, out=np.zeros_like(dlat), where=norm > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/knowair_joint_pm_weather_eda")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--analog-pairs", type=int, default=2000)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.root) / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    raw = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    timestamps = pd.date_range("2015-01-01", periods=len(raw), freq="3h")
    coords = np.loadtxt(root / "data/benchmarks/knowair/city.txt", usecols=(2, 3))
    stations = np.loadtxt(root / "data/benchmarks/knowair/city.txt", usecols=(1,), dtype=str)
    get = lambda name: np.asarray(raw[..., RAW.index(name)], dtype=np.float64)
    u, v = get("u100"), get("v100")
    pm_raw = get("pm25").copy(); pm_raw[pm_raw <= 0] = np.nan
    physical = {"temperature": get("temperature") - 273.15, "pressure": get("pressure") / 100,
                "rh950": get("rh950"), "wind_speed": np.hypot(u, v), "wind_u": u, "wind_v": v,
                "pbl": get("pbl"), "ventilation": np.hypot(u, v) * get("pbl"),
                "log_precipitation": np.log1p(np.maximum(get("precipitation") * 1000, 0))}
    variables = {"pm": climatology_anomaly(np.log1p(pm_raw), timestamps),
                 **{name: climatology_anomaly(values, timestamps) for name, values in physical.items()}}
    standardized = {name: values / np.maximum(np.nanstd(values, axis=0, keepdims=True), 1e-12)
                    for name, values in variables.items()}
    windows = len(raw) - 47

    # ------------------------------------------------------------------
    # Within-station directed cross-lag matrix with a negative-control view.
    cross_rows = []
    for lead_step in (1, 8, 16, 24):
        for source_name, source in variables.items():
            x = source[:-lead_step]
            for target_name, target in variables.items():
                y, target_now = target[lead_step:], target[:-lead_step]
                raw_r = station_corr(x, y)
                partial = (np.full(raw_r.shape, np.nan) if source_name == target_name
                           else station_partial(x, y, target_now))
                cross_rows.append({"past_feature": source_name, "future_feature": target_name,
                                   "lead_hour": lead_step * 3,
                                   **stat(raw_r, "raw_"), **stat(partial, "partial_control_target_now_")})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output / "within_station_cross_lag_matrix.csv", index=False)

    # Incremental predictive content and weather mediation of PM persistence.
    pm = variables["pm"]
    recent_pm = window_mean(pm, 16, 8, windows)
    previous_pm = window_mean(pm, 8, 8, windows)
    current_pm = pm[23:23 + windows]
    weather_last = {name: variables[name][23:23 + windows] for name in WEATHER_CORE}
    weather_recent = {name: window_mean(variables[name], 16, 8, windows) for name in WEATHER_CORE}
    weather_previous = {name: window_mean(variables[name], 8, 8, windows) for name in WEATHER_CORE}
    r2_rows, mediation_rows, station_gain_rows = [], [], []
    for lead_step in (1, 8, 16, 24):
        target = pm[23 + lead_step:23 + lead_step + windows]
        future_weather = {name: variables[name][23 + lead_step:23 + lead_step + windows]
                          for name in WEATHER_CORE}
        for station in range(len(stations)):
            base = np.column_stack([current_pm[:, station], recent_pm[:, station],
                                    recent_pm[:, station] - previous_pm[:, station]])
            past_extra = np.column_stack([
                array[:, station] for name in WEATHER_CORE
                for array in (weather_last[name], weather_recent[name],
                              weather_recent[name] - weather_previous[name])
            ])
            oracle_extra = np.column_stack([future_weather[name][:, station] for name in WEATHER_CORE])
            base_r2 = adjusted_r2(target[:, station], base)
            past_r2 = adjusted_r2(target[:, station], np.column_stack([base, past_extra]))
            oracle_r2 = adjusted_r2(target[:, station], np.column_stack([base, oracle_extra]))
            station_gain_rows.append({"station_index": station, "station": stations[station],
                                      "longitude": coords[station, 0], "latitude": coords[station, 1],
                                      "lead_hour": lead_step * 3, "pm_history_adjusted_r2": base_r2,
                                      "past_weather_r2_gain": past_r2 - base_r2,
                                      "realized_future_weather_r2_gain": oracle_r2 - base_r2})
        station_block = pd.DataFrame(station_gain_rows[-len(stations):])
        r2_rows.append({"lead_hour": lead_step * 3,
                        **stat(station_block.pm_history_adjusted_r2.to_numpy(), "pm_history_"),
                        **stat(station_block.past_weather_r2_gain.to_numpy(), "past_weather_gain_"),
                        **stat(station_block.realized_future_weather_r2_gain.to_numpy(), "future_weather_gain_")})
        # How much PM persistence remains after controlling future dispersion weather?
        uncontrolled, controlled = [], []
        for station in range(len(stations)):
            x, y = current_pm[:, station], target[:, station]
            controls = np.column_stack([future_weather[name][:, station]
                                        for name in ("ventilation", "rh950", "log_precipitation", "pressure")])
            uncontrolled.append(station_corr(x[:, None], y[:, None])[0])
            controlled.append(multiple_partial(x, y, controls))
        mediation_rows.append({"lead_hour": lead_step * 3,
                               **stat(np.asarray(uncontrolled), "raw_pm_persistence_"),
                               **stat(np.asarray(controlled), "after_future_weather_"),
                               "median_reduction": float(np.nanmedian(np.asarray(uncontrolled) - np.asarray(controlled)))})
    r2_frame = pd.DataFrame(r2_rows)
    mediation = pd.DataFrame(mediation_rows)
    station_gains = pd.DataFrame(station_gain_rows)
    r2_frame.to_csv(output / "pm_predictive_information_decomposition.csv", index=False)
    mediation.to_csv(output / "pm_persistence_weather_mediation.csv", index=False)
    station_gains.to_csv(output / "station_pm_weather_value_add.csv", index=False)

    # Geography of weather value-add.
    geography_rows = []
    for lead in (3, 24, 48, 72):
        block = station_gains[station_gains.lead_hour == lead]
        for metric in ("pm_history_adjusted_r2", "past_weather_r2_gain", "realized_future_weather_r2_gain"):
            geography_rows.append({"lead_hour": lead, "metric": metric,
                                   "spearman_with_latitude": float(spearmanr(block.latitude, block[metric]).statistic),
                                   "spearman_with_longitude": float(spearmanr(block.longitude, block[metric]).statistic)})
    pd.DataFrame(geography_rows).to_csv(output / "geographic_pm_weather_skill.csv", index=False)

    # ------------------------------------------------------------------
    # Joint analog determinism: PM-only, weather-only, and joint histories.
    rng = np.random.default_rng(args.seed)
    left, right = [], []
    while len(left) < args.analog_pairs:
        a, b = int(rng.integers(0, windows)), int(rng.integers(0, windows))
        if abs(a - b) >= 48:
            left.append(a); right.append(b)
    left, right = np.asarray(left), np.asarray(right)
    analog_rows = []
    for station in range(len(stations)):
        def distance(feature_names: tuple[str, ...], offset: int, length: int) -> np.ndarray:
            pieces = []
            for name in feature_names:
                x = np.nan_to_num(standardized[name][:, station])
                pieces.extend(np.square(x[left + offset + k] - x[right + offset + k]) for k in range(length))
            return np.sqrt(np.mean(pieces, axis=0))
        histories = {}
        for span, offset, length in (("recent24", 16, 8), ("full72", 0, 24)):
            histories[("pm_only", span)] = distance(("pm",), offset, length)
            histories[("weather_only", span)] = distance(MODE_WEATHER, offset, length)
            histories[("joint", span)] = distance(("pm",) + MODE_WEATHER, offset, length)
        futures = {}
        for horizon, offset, length in (("day1", 24, 8), ("day3", 40, 8), ("full72", 24, 24)):
            futures[("pm", horizon)] = distance(("pm",), offset, length)
            futures[("weather", horizon)] = distance(MODE_WEATHER, offset, length)
            futures[("joint", horizon)] = distance(("pm",) + MODE_WEATHER, offset, length)
        for (history_type, span), history_distance in histories.items():
            for (future_type, horizon), future_distance in futures.items():
                analog_rows.append({"station_index": station, "station": stations[station],
                                    "history_type": history_type, "history_span": span,
                                    "future_type": future_type, "future_horizon": horizon,
                                    "spearman_distance_rho": float(spearmanr(history_distance, future_distance).statistic)})
    analog = pd.DataFrame(analog_rows)
    analog.to_csv(output / "joint_analog_predictability.csv", index=False)

    # ------------------------------------------------------------------
    # Event composites: clean persistence, onset, polluted persistence, clearance.
    past_day_pm = window_mean(pm_raw, 16, 8, windows)
    future_day_pm = window_mean(pm_raw, 24, 8, windows)
    event_masks = {
        "clean_persistence": (past_day_pm <= 35) & (future_day_pm <= 35),
        "onset": (past_day_pm <= 35) & (future_day_pm > 75),
        "polluted_persistence": (past_day_pm > 75) & (future_day_pm > 75),
        "clearance": (past_day_pm > 75) & (future_day_pm <= 35),
    }
    composite_features = ("pm", "temperature", "pressure", "rh950", "wind_speed",
                          "pbl", "ventilation", "log_precipitation")
    composite_rows = []
    relative_indices = sorted(set(range(0, 48, 2)) | {47})  # every 6h plus exact +72h
    for feature in composite_features:
        values = standardized[feature]
        for relative_index in relative_indices:
            aligned = values[relative_index:relative_index + windows]
            relative_hour = -69 + relative_index * 3
            for event, mask in event_masks.items():
                valid = mask & np.isfinite(aligned)
                count = valid.sum(0)
                station_mean = np.divide(np.where(valid, aligned, 0).sum(0), count,
                                         out=np.full(len(stations), np.nan), where=count >= 8)
                composite_rows.append({"feature": feature, "event": event,
                                       "relative_hour": relative_hour,
                                       **stat(station_mean, "station_mean_"),
                                       "total_observations": int(valid.sum()),
                                       "median_events_per_station": float(np.median(count))})
    composites = pd.DataFrame(composite_rows)
    composites.to_csv(output / "joint_event_composites.csv", index=False)
    event_counts = []
    for event, mask in event_masks.items():
        counts = mask.sum(0)
        event_counts.append({"event": event, "total_station_origins": int(mask.sum()),
                             "median_per_station": float(np.median(counts)),
                             "q25_per_station": float(np.quantile(counts, .25)),
                             "q75_per_station": float(np.quantile(counts, .75))})
    pd.DataFrame(event_counts).to_csv(output / "event_counts.csv", index=False)

    # ------------------------------------------------------------------
    # Cross-station transport beyond target persistence and regional PM mode.
    distance_km, east, north = spatial_geometry(coords)
    nearest = np.argsort(distance_km, axis=1)[:, 1:6]
    regional_pm = np.nanmean(pm, axis=1)
    edge_rows, neighbour_gain_rows = [], []
    for target_station in range(len(stations)):
        sources = nearest[target_station]
        source_speed = np.hypot(u[:, sources], v[:, sources])
        alignment = ((u[:, sources] * east[sources, target_station]
                      + v[:, sources] * north[sources, target_station]) / np.maximum(source_speed, 1e-12))
        positive = np.maximum(alignment, 0)
        source_pm = pm[:, sources]
        weight = np.nansum(positive, axis=1)
        aligned_neighbor = np.divide(np.nansum(source_pm * positive, axis=1), weight,
                                     out=np.full(len(pm), np.nan), where=weight > 1e-12)
        unweighted_neighbor = np.nanmean(source_pm, axis=1)
        for lead_step in (1, 2, 4, 8):
            y = pm[lead_step:, target_station]
            base_controls = np.column_stack([pm[:-lead_step, target_station], regional_pm[:-lead_step]])
            base_r2 = adjusted_r2(y, base_controls)
            aligned_r2 = adjusted_r2(y, np.column_stack([base_controls, aligned_neighbor[:-lead_step]]))
            unweighted_r2 = adjusted_r2(y, np.column_stack([base_controls, unweighted_neighbor[:-lead_step]]))
            neighbour_gain_rows.append({"target_index": target_station, "target": stations[target_station],
                                        "lead_hour": lead_step * 3, "base_adjusted_r2": base_r2,
                                        "aligned_neighbor_r2_gain": aligned_r2 - base_r2,
                                        "unweighted_neighbor_r2_gain": unweighted_r2 - base_r2})
            for source_position, source_station in enumerate(sources):
                x = pm[:-lead_step, source_station]
                target_now = pm[:-lead_step, target_station]
                regional_now = regional_pm[:-lead_step]
                future_controls = np.column_stack([
                    target_now, regional_now,
                    variables["ventilation"][lead_step:, target_station],
                    variables["rh950"][lead_step:, target_station],
                    variables["log_precipitation"][lead_step:, target_station],
                ])
                align = alignment[:-lead_step, source_position]
                for regime, select in (("downwind", align >= .5), ("crosswind", np.abs(align) < .5),
                                       ("upwind", align <= -.5)):
                    basic = multiple_partial(x, y, target_now[:, None], select)
                    regional_control = multiple_partial(x, y, np.column_stack([target_now, regional_now]), select)
                    weather_control = multiple_partial(x, y, future_controls, select)
                    edge_rows.append({"source_index": int(source_station), "source": stations[source_station],
                                      "target_index": target_station, "target": stations[target_station],
                                      "distance_km": distance_km[source_station, target_station],
                                      "lead_hour": lead_step * 3, "wind_regime": regime,
                                      "partial_r_control_target_pm": basic,
                                      "partial_r_control_target_and_regional_pm": regional_control,
                                      "partial_r_add_future_target_weather_controls": weather_control})
    edges = pd.DataFrame(edge_rows)
    neighbour_gains = pd.DataFrame(neighbour_gain_rows)
    edges.to_csv(output / "cross_station_joint_transport_edges.csv", index=False)
    neighbour_gains.to_csv(output / "dynamic_neighbor_value_add.csv", index=False)
    edge_summary = (edges.groupby(["lead_hour", "wind_regime"])
                    .agg(edges=("partial_r_control_target_pm", "count"),
                         median_target_control=("partial_r_control_target_pm", "median"),
                         median_regional_control=("partial_r_control_target_and_regional_pm", "median"),
                         median_weather_control=("partial_r_add_future_target_weather_controls", "median"),
                         median_distance_km=("distance_km", "median"))
                    .reset_index())
    edge_summary.to_csv(output / "cross_station_joint_transport_summary.csv", index=False)

    # Coupled regional modes: do past/future regional weather modes add to PM modes?
    pm_scores, pm_explained = pca_scores(pm, 5)
    weather_scores = []
    weather_explained = {}
    for name in MODE_WEATHER:
        scores, explained = pca_scores(variables[name], 3)
        weather_scores.append(scores)
        weather_explained[name] = explained.tolist()
    weather_scores = np.column_stack(weather_scores)
    mode_rows = []
    for lead_step in (1, 8, 16, 24):
        y = pm_scores[lead_step:]
        base, past_w, future_w = pm_scores[:-lead_step], weather_scores[:-lead_step], weather_scores[lead_step:]
        for component in range(5):
            base_r2 = adjusted_r2(y[:, component], base)
            past_r2 = adjusted_r2(y[:, component], np.column_stack([base, past_w]))
            future_r2 = adjusted_r2(y[:, component], np.column_stack([base, future_w]))
            mode_rows.append({"pm_component": component + 1, "lead_hour": lead_step * 3,
                              "pm_mode_explained_variance": pm_explained[component],
                              "pm_history_adjusted_r2": base_r2,
                              "past_weather_mode_r2_gain": past_r2 - base_r2,
                              "future_weather_mode_r2_gain": future_r2 - base_r2})
    modes = pd.DataFrame(mode_rows)
    modes.to_csv(output / "regional_pm_weather_mode_coupling.csv", index=False)

    # Figures.
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    heat = cross[cross.lead_hour == 24].pivot(index="past_feature", columns="future_feature",
                                             values="partial_control_target_now_median")
    sns.heatmap(heat, cmap="vlag", center=0, vmin=-.35, vmax=.35, ax=axes[0])
    axes[0].set_title("Past -> future incremental coupling (+24h)")
    r2_long = r2_frame.melt(id_vars="lead_hour",
                            value_vars=["pm_history_median", "past_weather_gain_median", "future_weather_gain_median"],
                            var_name="information", value_name="adjusted_r2_or_gain")
    sns.lineplot(data=r2_long, x="lead_hour", y="adjusted_r2_or_gain", hue="information", marker="o", ax=axes[1])
    axes[1].set_title("PM predictive information decomposition")
    sns.lineplot(data=mediation, x="lead_hour", y="raw_pm_persistence_median", marker="o", label="raw", ax=axes[2])
    sns.lineplot(data=mediation, x="lead_hour", y="after_future_weather_median", marker="o",
                 label="after future weather", ax=axes[2])
    axes[2].set_title("Weather-mediated PM persistence")
    fig.savefig(output / "within_station_joint_information.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, ("pm", "ventilation", "pressure", "rh950")):
        subset = composites[composites.feature == feature]
        sns.lineplot(data=subset, x="relative_hour", y="station_mean_median", hue="event", ax=axis)
        axis.axvline(0, color="black", lw=.8)
        axis.set(title=LABELS[feature], ylabel="Median station standardized anomaly")
    fig.suptitle("Joint PM-weather event trajectories", fontsize=16)
    fig.savefig(output / "joint_event_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    sns.barplot(data=edge_summary, x="lead_hour", y="median_regional_control", hue="wind_regime", ax=axes[0])
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set(title="Source PM value-add beyond target + regional PM", ylabel="Median partial r")
    gain_summary = neighbour_gains.groupby("lead_hour")[["aligned_neighbor_r2_gain", "unweighted_neighbor_r2_gain"]].median().reset_index()
    gain_long = gain_summary.melt(id_vars="lead_hour", var_name="neighbor", value_name="adjusted_r2_gain")
    sns.barplot(data=gain_long, x="lead_hour", y="adjusted_r2_gain", hue="neighbor", ax=axes[1])
    axes[1].set(title="Dynamic neighbour PM value-add", ylabel="Median adjusted R2 gain")
    fig.savefig(output / "cross_station_joint_transport.png", dpi=180)
    plt.close(fig)

    analog_summary = (analog.groupby(["history_type", "history_span", "future_type", "future_horizon"])
                      .spearman_distance_rho.median().rename("median_rho").reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    analog_pm = analog_summary[(analog_summary.future_type == "pm") & (analog_summary.history_span == "recent24")]
    sns.barplot(data=analog_pm, x="future_horizon", y="median_rho", hue="history_type", ax=axes[0])
    axes[0].set_title("Which past state identifies future PM analogs?")
    mode1 = modes[modes.pm_component == 1]
    sns.lineplot(data=mode1, x="lead_hour", y="pm_history_adjusted_r2", marker="o", label="PM modes", ax=axes[1])
    sns.lineplot(data=mode1, x="lead_hour", y="past_weather_mode_r2_gain", marker="o", label="+ past weather", ax=axes[1])
    sns.lineplot(data=mode1, x="lead_hour", y="future_weather_mode_r2_gain", marker="o", label="+ future weather", ax=axes[1])
    axes[1].set_title("Regional PM mode information")
    fig.savefig(output / "joint_analogs_and_regional_modes.png", dpi=180)
    plt.close(fig)

    summary_payload = {
        "dataset": {"time_steps": len(raw), "stations": len(stations), "overlapping_windows": windows,
                    "pm_nonpositive_treated_missing_fraction": float(np.mean(~np.isfinite(pm_raw))),
                    "split_boundaries_used": False},
        "pm_information": r2_frame.to_dict("records"),
        "pm_weather_mediation": mediation.to_dict("records"),
        "event_counts": event_counts,
        "transport_summary": edge_summary.to_dict("records"),
        "analog_summary": analog_summary.to_dict("records"),
        "regional_pm_explained": pm_explained.tolist(),
        "regional_weather_explained": weather_explained,
    }
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(list(output.iterdir())),
                      "cross_lag_cells": len(cross), "transport_edges": len(edges),
                      "analog_rows": len(analog)}, indent=2))


if __name__ == "__main__":
    main()
