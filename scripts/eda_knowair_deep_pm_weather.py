"""Deep, split-agnostic EDA of PM2.5 and weather dynamics in KnowAir.

Every valid 72h-history/72h-future origin is retained, including overlapping
windows. PM <= 0 is treated as missing. Dynamic correlations use log1p(PM)
anomalies after removing station x month x dataset-clock-hour climatology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAW = (
    "u100", "v100", "dewpoint", "temperature", "pbl", "k_index", "rh950",
    "rh975", "specific_humidity950", "pressure", "temperature925",
    "temperature950", "precipitation", "u950", "v950", "vertical_velocity950",
    "vorticity950", "pm25",
)
FEATURE_LABELS = {
    "temperature": "Temperature", "pressure": "Pressure", "rh950": "RH 950 hPa",
    "wind_speed": "Wind speed", "wind_u": "Wind u", "wind_v": "Wind v",
    "pbl": "Boundary-layer height", "ventilation": "Ventilation",
    "dewpoint_deficit": "Dewpoint deficit", "precipitation": "Precipitation",
    "pm": "PM2.5",
}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def station_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise-valid Pearson correlation down time for every station."""
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(0)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    sx, sy = xv.sum(0), yv.sum(0)
    sxx, syy = np.square(xv).sum(0), np.square(yv).sum(0)
    sxy = (xv * yv).sum(0)
    cov = sxy - sx * sy / np.maximum(count, 1)
    vx = sxx - sx * sx / np.maximum(count, 1)
    vy = syy - sy * sy / np.maximum(count, 1)
    denominator = np.sqrt(np.maximum(vx * vy, 0.0))
    return np.divide(cov, denominator, out=np.full(x.shape[1], np.nan),
                     where=(count >= 20) & (denominator > 1e-12))


def corr_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    values = station_corr(x, y)
    return {"median_r": float(np.nanmedian(values)),
            "q25_r": float(np.nanquantile(values, .25)),
            "q75_r": float(np.nanquantile(values, .75)),
            "stations": int(np.isfinite(values).sum())}


def partial_corr_summary(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> dict[str, float]:
    """Station-wise partial r(x,y|control), summarized across stations."""
    rxy, rxz, ryz = station_corr(x, y), station_corr(x, control), station_corr(y, control)
    denominator = np.sqrt(np.maximum((1 - rxz * rxz) * (1 - ryz * ryz), 0.0))
    partial = np.divide(rxy - rxz * ryz, denominator, out=np.full_like(rxy, np.nan),
                        where=denominator > 1e-12)
    return {"median_partial_r": float(np.nanmedian(partial)),
            "q25_partial_r": float(np.nanquantile(partial, .25)),
            "q75_partial_r": float(np.nanquantile(partial, .75))}


def climatology_anomaly(values: np.ndarray, timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return anomalies and time-aligned station/month/hour climatology."""
    anomaly = np.full(values.shape, np.nan, dtype=np.float64)
    baseline = np.full(values.shape, np.nan, dtype=np.float64)
    groups = timestamps.month.to_numpy() * 100 + timestamps.hour.to_numpy()
    for group in np.unique(groups):
        mask = groups == group
        block = np.asarray(values[mask], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=0, keepdims=True)
        baseline[mask] = mean
        anomaly[mask] = block - mean
    return anomaly, baseline


def window_mean(values: np.ndarray, start: int, length: int, windows: int,
                minimum_fraction: float = .75) -> np.ndarray:
    valid = np.isfinite(values)
    cumulative = np.concatenate([np.zeros((1, values.shape[1])),
                                 np.cumsum(np.where(valid, values, 0.0), axis=0)], axis=0)
    counts = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.int32),
                             np.cumsum(valid, axis=0, dtype=np.int32)], axis=0)
    total = cumulative[start + length:start + length + windows] - cumulative[start:start + windows]
    count = counts[start + length:start + length + windows] - counts[start:start + windows]
    return np.divide(total, count, out=np.full(total.shape, np.nan),
                     where=count >= int(np.ceil(length * minimum_fraction)))


def quantile_bins(values: np.ndarray, bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    edges = np.quantile(finite, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = np.full(values.shape, -1, dtype=np.int8)
    labels[np.isfinite(values)] = np.searchsorted(edges[1:-1], values[np.isfinite(values)], side="right")
    return labels, edges


def category_pm(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid] = np.digitize(values[valid], [35.0, 75.0, 150.0])
    return result


def partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray, select: np.ndarray) -> float:
    valid = select & np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if valid.sum() < 80:
        return np.nan
    matrix = np.corrcoef(np.stack([x[valid], y[valid], control[valid]]))
    rxy, rxz, ryz = matrix[0, 1], matrix[0, 2], matrix[1, 2]
    denominator = np.sqrt(max((1 - rxz * rxz) * (1 - ryz * ryz), 1e-12))
    return float((rxy - rxz * ryz) / denominator)


def haversine_and_unit_vectors(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlon = lon[None, :] - lon[:, None]
    dlat = lat[None, :] - lat[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    east = dlon * np.cos((lat[:, None] + lat[None, :]) / 2)
    north = dlat
    norm = np.hypot(east, north)
    east = np.divide(east, norm, out=np.zeros_like(east), where=norm > 0)
    north = np.divide(north, norm, out=np.zeros_like(north), where=norm > 0)
    return distance, east, north


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/knowair_deep_pm_weather_eda")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.root) / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    raw = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    timestamps = pd.date_range("2015-01-01", periods=len(raw), freq="3h")
    steps, windows = 24, len(raw) - 47
    get = lambda name: np.asarray(raw[..., RAW.index(name)], dtype=np.float64)
    u, v = get("u100"), get("v100")
    pm = get("pm25").copy()
    pm[pm <= 0] = np.nan
    log_pm = np.log1p(pm)
    pm_anomaly, pm_climatology = climatology_anomaly(log_pm, timestamps)
    physical = {
        "temperature": get("temperature") - 273.15,
        "pressure": get("pressure") / 100.0,
        "rh950": get("rh950"), "wind_speed": np.hypot(u, v),
        "wind_u": u, "wind_v": v, "pbl": get("pbl"),
        "ventilation": np.hypot(u, v) * get("pbl"),
        "dewpoint_deficit": get("temperature") - get("dewpoint"),
        "precipitation": get("precipitation") * 1000.0,
    }
    anomaly = {name: climatology_anomaly(values, timestamps)[0]
               for name, values in physical.items()}

    # Exact overlapping-window summaries.
    past_pm_72 = window_mean(pm, 0, 24, windows)
    past_log_72 = window_mean(pm_anomaly, 0, 24, windows)
    past_pm_recent24 = window_mean(pm, 16, 8, windows)
    past_log_recent24 = window_mean(pm_anomaly, 16, 8, windows)
    past_log_previous24 = window_mean(pm_anomaly, 8, 8, windows)
    future_pm_days = [window_mean(pm, 24 + 8 * day, 8, windows) for day in range(3)]
    future_log_days = [window_mean(pm_anomaly, 24 + 8 * day, 8, windows) for day in range(3)]
    future_pm_72 = window_mean(pm, 24, 24, windows)
    future_log_72 = window_mean(pm_anomaly, 24, 24, windows)
    last_log = pm_anomaly[23:23 + windows]
    last_pm = pm[23:23 + windows]

    # PM persistence by predictor and lead.
    persistence_rows = []
    for lead_step in range(1, 25):
        future_log = pm_anomaly[23 + lead_step:23 + lead_step + windows]
        future_raw = pm[23 + lead_step:23 + lead_step + windows]
        for predictor, values in (("last", last_log), ("past_24h_mean", past_log_recent24),
                                  ("past_72h_mean", past_log_72)):
            persistence_rows.append({"lead_hour": lead_step * 3, "predictor": predictor,
                                     **corr_summary(values, future_log)})
        valid = np.isfinite(last_pm) & np.isfinite(future_raw)
        absolute_error = np.where(valid, np.abs(future_raw - last_pm), np.nan)
        persistence_rows[-3]["persistence_mae_ugm3"] = float(np.nanmedian(np.nanmean(absolute_error, axis=0)))
    persistence = pd.DataFrame(persistence_rows)
    persistence.to_csv(output / "pm_persistence_by_lead.csv", index=False)

    # Cross-lag feature -> future PM, contrasting historical and realized future weather.
    cross_rows = []
    predictors = {"pm": pm_anomaly, **anomaly}
    for feature, values in predictors.items():
        past_last = values[23:23 + windows]
        past_mean = window_mean(values, 0, 24, windows)
        for lead_step in range(1, 25):
            target = pm_anomaly[23 + lead_step:23 + lead_step + windows]
            for timing, source in (("past_last", past_last), ("past_72h_mean", past_mean)):
                cross_rows.append({"feature": feature, "timing": timing,
                                   "lead_hour": lead_step * 3, **corr_summary(source, target)})
            if feature != "pm":
                future_same = values[23 + lead_step:23 + lead_step + windows]
                cross_rows.append({"feature": feature, "timing": "future_same_time",
                                   "lead_hour": lead_step * 3, **corr_summary(future_same, target)})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output / "feature_to_future_pm_correlations.csv", index=False)

    # Mean reversion conditional on the recent PM state.
    state_bins, state_edges = quantile_bins(past_log_recent24)
    reversion_rows = []
    for lead_step in (1, 8, 16, 24):
        future_raw = pm[23 + lead_step:23 + lead_step + windows]
        future_log = pm_anomaly[23 + lead_step:23 + lead_step + windows]
        for state in range(5):
            select = (state_bins == state) & np.isfinite(last_pm) & np.isfinite(future_raw)
            reversion_rows.append({
                "lead_hour": lead_step * 3, "past_pm_anomaly_quintile": state + 1,
                "n": int(select.sum()), "median_last_pm": float(np.nanmedian(last_pm[select])),
                "median_future_pm": float(np.nanmedian(future_raw[select])),
                "median_change_pm": float(np.nanmedian((future_raw - last_pm)[select])),
                "median_future_log_anomaly": float(np.nanmedian(future_log[select])),
            })
    reversion = pd.DataFrame(reversion_rows)
    reversion.to_csv(output / "pm_mean_reversion_by_state.csv", index=False)

    # Trend continuation versus reversal over the three future days.
    trend = past_log_recent24 - past_log_previous24
    trend_bins, trend_edges = quantile_bins(trend)
    trend_rows = []
    for day, future_day in enumerate(future_log_days, start=1):
        future_change = future_day - past_log_recent24
        change_summary = corr_summary(trend, future_change)
        level_summary = corr_summary(trend, future_day)
        incremental_summary = partial_corr_summary(trend, future_day, past_log_recent24)
        valid = np.isfinite(trend) & np.isfinite(future_change) & (np.abs(trend) > 1e-12)
        trend_rows.append({"future_day": day, "trend_quintile": "all",
                           "median_r_with_future_change": change_summary["median_r"],
                           "median_r_with_future_level": level_summary["median_r"],
                           **incremental_summary,
                           "same_sign_probability": float(np.mean(np.sign(trend[valid]) == np.sign(future_change[valid]))),
                           "n": int(valid.sum())})
        for q in range(5):
            select = valid & (trend_bins == q)
            trend_rows.append({"future_day": day, "trend_quintile": q + 1,
                               "median_past_trend": float(np.nanmedian(trend[select])),
                               "median_future_change": float(np.nanmedian(future_change[select])),
                               "same_sign_probability": float(np.mean(np.sign(trend[select]) == np.sign(future_change[select]))),
                               "n": int(select.sum())})
    trend_frame = pd.DataFrame(trend_rows)
    trend_frame.to_csv(output / "pm_trend_continuation.csv", index=False)

    # Pollution-category transitions for each future 24-hour block.
    names = ("<=35", "35-75", "75-150", ">150")
    past_category = category_pm(past_pm_recent24)
    transition_rows = []
    for day, future_day in enumerate(future_pm_days, start=1):
        future_category = category_pm(future_day)
        for source in range(4):
            denominator = int((past_category == source).sum())
            for destination in range(4):
                count = int(((past_category == source) & (future_category == destination)).sum())
                transition_rows.append({"future_day": day, "past_category": names[source],
                                        "future_category": names[destination], "n": count,
                                        "probability": count / denominator if denominator else np.nan})
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(output / "pm_category_transitions.csv", index=False)

    # Episode duration at common thresholds; missing values break a run.
    episode_rows = []
    for threshold in (35.0, 75.0, 150.0):
        durations = []
        for station in range(pm.shape[1]):
            active = np.isfinite(pm[:, station]) & (pm[:, station] > threshold)
            padded = np.r_[False, active, False].astype(np.int8)
            changes = np.diff(padded)
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            durations.extend(((ends - starts) * 3).tolist())
        durations = np.asarray(durations)
        episode_rows.append({"threshold_ugm3": threshold, "episodes": len(durations),
                             "median_hours": float(np.median(durations)),
                             "p75_hours": float(np.quantile(durations, .75)),
                             "p90_hours": float(np.quantile(durations, .90)),
                             "p95_hours": float(np.quantile(durations, .95)),
                             "max_hours": int(durations.max()),
                             "probability_over_24h": float(np.mean(durations > 24)),
                             "probability_over_48h": float(np.mean(durations > 48))})
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(output / "pm_episode_durations.csv", index=False)

    # Joint PM-state x realized-future-weather regimes.
    regime_features = ("wind_speed", "pbl", "ventilation", "rh950", "precipitation")
    regime_rows = []
    for feature in regime_features:
        future_weather = window_mean(anomaly[feature], 24, 24, windows)
        weather_bins, weather_edges = quantile_bins(future_weather)
        for pm_q in range(5):
            for weather_q in range(5):
                select = ((state_bins == pm_q) & (weather_bins == weather_q)
                          & np.isfinite(past_pm_72) & np.isfinite(future_pm_72))
                regime_rows.append({
                    "weather_feature": feature, "past_pm_quintile": pm_q + 1,
                    "future_weather_quintile": weather_q + 1, "n": int(select.sum()),
                    "median_past_pm": float(np.nanmedian(past_pm_72[select])),
                    "median_future_pm": float(np.nanmedian(future_pm_72[select])),
                    "median_pm_change": float(np.nanmedian((future_pm_72 - past_pm_72)[select])),
                    "median_future_log_anomaly": float(np.nanmedian(future_log_72[select])),
                    "prob_future_mean_over_75": float(np.mean(future_pm_72[select] > 75)),
                    "prob_future_mean_over_150": float(np.mean(future_pm_72[select] > 150)),
                })
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(output / "joint_pm_weather_regimes.csv", index=False)

    # Raw accumulated precipitation is zero-inflated, so report interpretable
    # physical bins in addition to anomaly quintiles.
    future_rain_total = window_mean(physical["precipitation"], 24, 24, windows, 1.0) * 24
    rain_edges = (-np.inf, .1, 1.0, 5.0, np.inf)
    rain_names = ("<=0.1mm", "0.1-1mm", "1-5mm", ">5mm")
    rain_bin = np.digitize(future_rain_total, rain_edges[1:-1])
    rain_rows = []
    for pm_q in range(5):
        for rain_q, rain_name in enumerate(rain_names):
            select = ((state_bins == pm_q) & (rain_bin == rain_q)
                      & np.isfinite(past_pm_72) & np.isfinite(future_pm_72))
            rain_rows.append({"past_pm_quintile": pm_q + 1, "future_72h_rain": rain_name,
                              "n": int(select.sum()),
                              "median_rain_mm": float(np.nanmedian(future_rain_total[select])),
                              "median_pm_change": float(np.nanmedian((future_pm_72 - past_pm_72)[select])),
                              "prob_future_mean_over_75": float(np.mean(future_pm_72[select] > 75))})
    rain_regimes = pd.DataFrame(rain_rows)
    rain_regimes.to_csv(output / "physical_precipitation_regimes.csv", index=False)

    # Seasonal stability of PM persistence.
    origin_times = timestamps[23:23 + windows]
    season_labels = np.array([SEASONS[month] for month in origin_times.month])
    seasonal_rows = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        origin_select = season_labels == season
        for lead_step in (1, 8, 16, 24):
            future = pm_anomaly[23 + lead_step:23 + lead_step + windows]
            seasonal_rows.append({"season": season, "lead_hour": lead_step * 3,
                                  **corr_summary(last_log[origin_select], future[origin_select])})
    seasonal = pd.DataFrame(seasonal_rows)
    seasonal.to_csv(output / "seasonal_pm_persistence.csv", index=False)

    # Absolute PM climatology and year-to-year robustness are kept separate from
    # anomaly dynamics so long-range seasonal structure is not mistaken for skill.
    temporal_rows = []
    for month in range(1, 13):
        for hour in sorted(np.unique(timestamps.hour)):
            select = (timestamps.month == month) & (timestamps.hour == hour)
            values = pm[select]
            temporal_rows.append({"month": month, "dataset_clock_hour": int(hour),
                                  "n": int(np.isfinite(values).sum()),
                                  "median_pm": float(np.nanmedian(values)),
                                  "p90_pm": float(np.nanquantile(values, .9)),
                                  "prob_pm_over_75": float(np.nanmean(np.where(np.isfinite(values), values > 75, np.nan))),
                                  "prob_pm_over_150": float(np.nanmean(np.where(np.isfinite(values), values > 150, np.nan)))})
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(output / "pm_month_hour_climatology.csv", index=False)

    yearly_rows = []
    for year in range(2015, 2019):
        origin_select = origin_times.year == year
        for lead_step in (1, 8, 16, 24):
            target = pm_anomaly[23 + lead_step:23 + lead_step + windows]
            yearly_rows.append({"year": year, "diagnostic": "pm_last_persistence",
                                "lead_hour": lead_step * 3,
                                **corr_summary(last_log[origin_select], target[origin_select])})
            future_ventilation = anomaly["ventilation"][23 + lead_step:23 + lead_step + windows]
            yearly_rows.append({"year": year, "diagnostic": "future_ventilation_vs_pm",
                                "lead_hour": lead_step * 3,
                                **corr_summary(future_ventilation[origin_select], target[origin_select])})
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(output / "yearly_robustness.csv", index=False)

    # Direction-conditioned nearest-neighbour transport diagnostic. Partial r
    # controls for the target station's current PM to isolate neighbour value-add.
    coords = np.loadtxt(root / "data/benchmarks/knowair/city.txt", usecols=(2, 3))
    distance, east, north = haversine_and_unit_vectors(coords)
    nearest = np.argsort(distance, axis=1)[:, 1:6]
    spatial_rows = []
    for target in range(pm.shape[1]):
        for source in nearest[target]:
            speed = np.hypot(u[:, source], v[:, source])
            alignment = (u[:, source] * east[source, target] + v[:, source] * north[source, target]) / np.maximum(speed, 1e-12)
            for lead_step in (1, 2, 4, 8):
                x = pm_anomaly[:-lead_step, source]
                y = pm_anomaly[lead_step:, target]
                control = pm_anomaly[:-lead_step, target]
                align = alignment[:-lead_step]
                for regime, select in (("downwind", align >= .5), ("crosswind", np.abs(align) < .5),
                                       ("upwind", align <= -.5)):
                    spatial_rows.append({
                        "source": int(source), "target": int(target),
                        "distance_km": float(distance[source, target]),
                        "lead_hour": lead_step * 3, "wind_regime": regime,
                        "n": int((select & np.isfinite(x) & np.isfinite(y) & np.isfinite(control)).sum()),
                        "partial_r_source_to_target_given_target_now": partial_corr(x, y, control, select),
                    })
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(output / "directional_spatial_transport.csv", index=False)
    spatial_summary = (spatial.groupby(["lead_hour", "wind_regime"])
                       .agg(edges=("partial_r_source_to_target_given_target_now", "count"),
                            median_partial_r=("partial_r_source_to_target_given_target_now", "median"),
                            q25=("partial_r_source_to_target_given_target_now", lambda x: x.quantile(.25)),
                            q75=("partial_r_source_to_target_given_target_now", lambda x: x.quantile(.75)),
                            median_n=("n", "median"), median_distance_km=("distance_km", "median"))
                       .reset_index())
    spatial_summary.to_csv(output / "directional_spatial_transport_summary.csv", index=False)
    spatial["distance_bin"] = pd.cut(spatial.distance_km, [0, 50, 100, 200, np.inf],
                                     labels=["<50", "50-100", "100-200", ">=200"], right=False)
    spatial_distance_summary = (spatial.groupby(["distance_bin", "lead_hour", "wind_regime"], observed=True)
                                .agg(edges=("partial_r_source_to_target_given_target_now", "count"),
                                     median_partial_r=("partial_r_source_to_target_given_target_now", "median"),
                                     q25=("partial_r_source_to_target_given_target_now", lambda x: x.quantile(.25)),
                                     q75=("partial_r_source_to_target_given_target_now", lambda x: x.quantile(.75)))
                                .reset_index())
    spatial_distance_summary.to_csv(output / "directional_transport_by_distance.csv", index=False)

    # Figures.
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for predictor in ("last", "past_24h_mean", "past_72h_mean"):
        line = persistence[persistence.predictor == predictor]
        axes[0].plot(line.lead_hour, line.median_r, marker="o", ms=3, label=predictor)
        axes[0].fill_between(line.lead_hour, line.q25_r, line.q75_r, alpha=.10)
    axes[0].set(title="PM dynamic persistence (log anomaly)", xlabel="Lead (hours)", ylabel="Median station r", ylim=(-.2, 1))
    axes[0].legend()
    sns.lineplot(data=reversion, x="lead_hour", y="median_change_pm", hue="past_pm_anomaly_quintile",
                 palette="coolwarm", marker="o", ax=axes[1])
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(title="Regression to the mean by recent PM state", xlabel="Lead (hours)", ylabel="Median PM change (ug/m3)")
    fig.savefig(output / "pm_persistence_and_mean_reversion.png", dpi=180)
    plt.close(fig)

    selected_features = ("temperature", "pressure", "rh950", "wind_speed", "pbl", "ventilation", "precipitation")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    for axis, feature in zip(axes.flat, selected_features):
        subset = cross[cross.feature == feature]
        for timing, style in (("past_last", "-"), ("past_72h_mean", "--"), ("future_same_time", ":")):
            line = subset[subset.timing == timing]
            axis.plot(line.lead_hour, line.median_r, style, lw=2, label=timing)
        axis.axhline(0, color="black", lw=.7)
        axis.set(title=FEATURE_LABELS[feature], xlabel="Lead (hours)", ylabel="Median station r")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Weather anomaly association with future PM anomaly", fontsize=16)
    fig.savefig(output / "weather_to_future_pm_by_lead.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for axis, feature in zip(axes.flat, regime_features):
        subset = regimes[regimes.weather_feature == feature]
        matrix = subset.pivot(index="past_pm_quintile", columns="future_weather_quintile", values="median_pm_change")
        bound = np.nanquantile(np.abs(matrix), .95)
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="vlag", center=0, vmin=-bound, vmax=bound,
                    ax=axis, cbar_kws={"label": "PM change (ug/m3)"})
        axis.set(title=FEATURE_LABELS[feature], xlabel="Future weather anomaly quintile", ylabel="Past PM anomaly quintile")
    axes.flat[-1].axis("off")
    fig.suptitle("72h PM change by initial pollution state and realized future weather", fontsize=16)
    fig.savefig(output / "joint_pm_weather_regime_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    sns.lineplot(data=seasonal, x="lead_hour", y="median_r", hue="season", marker="o", ax=axes[0])
    axes[0].set(title="PM persistence by season", ylabel="Median station r", ylim=(-.1, 1))
    sns.barplot(data=spatial_summary, x="lead_hour", y="median_partial_r", hue="wind_regime", ax=axes[1])
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(title="Neighbour PM value-add conditional on wind", ylabel="Median edge partial r")
    fig.savefig(output / "seasonal_and_spatial_diagnostics.png", dpi=180)
    plt.close(fig)

    downwind_distance = spatial_distance_summary[spatial_distance_summary.wind_regime == "downwind"]
    fig, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    sns.lineplot(data=downwind_distance, x="lead_hour", y="median_partial_r", hue="distance_bin",
                 marker="o", ax=axis)
    axis.set(title="Downwind neighbour signal shifts with distance",
             xlabel="Lead (hours)", ylabel="Median edge partial r")
    fig.savefig(output / "downwind_transport_by_distance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    median_matrix = temporal.pivot(index="month", columns="dataset_clock_hour", values="median_pm")
    exceed_matrix = temporal.pivot(index="month", columns="dataset_clock_hour", values="prob_pm_over_75")
    sns.heatmap(median_matrix, cmap="mako_r", annot=True, fmt=".0f", ax=axes[0],
                cbar_kws={"label": "Median PM (ug/m3)"})
    sns.heatmap(exceed_matrix, cmap="rocket_r", annot=True, fmt=".2f", ax=axes[1],
                cbar_kws={"label": "P(PM > 75)"})
    axes[0].set_title("Absolute PM climatology")
    axes[1].set_title("Pollution exceedance climatology")
    for axis in axes:
        axis.set(xlabel="Dataset-clock hour", ylabel="Month")
    fig.savefig(output / "pm_month_hour_climatology.png", dpi=180)
    plt.close(fig)

    key_leads = persistence[(persistence.predictor == "last") & persistence.lead_hour.isin([3, 24, 48, 72])]
    summary = {
        "dataset": {"time_steps": len(raw), "stations": raw.shape[1], "overlapping_windows": windows,
                    "pm_nonpositive_fraction_treated_missing": float(np.mean(~np.isfinite(pm))),
                    "split_boundaries_used": False},
        "pm_last_value_dynamic_correlation": {
            str(int(row.lead_hour)): float(row.median_r) for row in key_leads.itertuples()
        },
        "episode_duration": episode_rows,
        "trend_all": trend_frame[trend_frame.trend_quintile == "all"]
                     .dropna(axis=1, how="all").to_dict("records"),
        "spatial_transport": spatial_summary.to_dict("records"),
        "quantile_inner_edges": {"past_pm_log_anomaly": state_edges[1:-1].tolist(),
                                 "past_trend": trend_edges[1:-1].tolist()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
