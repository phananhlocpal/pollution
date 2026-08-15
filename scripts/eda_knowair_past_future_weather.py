"""EDA of 72-hour historical versus 72-hour future weather in KnowAir.

The analysis deliberately ignores train/validation/test boundaries.  Every valid
3-hourly origin is used, so adjacent (and therefore overlapping) windows are
included.  Correlations are summarized across stations to avoid a few spatially
variable sites dominating a single pooled statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAW_FEATURES = (
    "100m_u_wind", "100m_v_wind", "2m_dewpoint", "2m_temperature",
    "boundary_layer_height", "k_index", "relative_humidity_950",
    "relative_humidity_975", "specific_humidity_950", "surface_pressure",
    "temperature_925", "temperature_950", "total_precipitation",
    "u_wind_950", "v_wind_950", "vertical_velocity_950",
    "vorticity_950", "PM2.5",
)

FEATURE_LABELS = {
    "temperature_c": "Temperature (C)",
    "pressure_hpa": "Pressure (hPa)",
    "rh950_percent": "RH 950 hPa (%)",
    "wind_speed_ms": "100 m wind speed (m/s)",
    "wind_u_ms": "100 m zonal wind u (m/s)",
    "wind_v_ms": "100 m meridional wind v (m/s)",
}


def station_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation down time, independently for every station."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    denominator = np.sqrt(np.square(xc).sum(0) * np.square(yc).sum(0))
    return np.divide(
        (xc * yc).sum(0), denominator,
        out=np.full(x.shape[1], np.nan), where=denominator > 1e-12,
    )


def summarize_correlations(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanquantile(values, 0.25)),
        "q75": float(np.nanquantile(values, 0.75)),
    }


def remove_station_month_hour_climatology(
    values: np.ndarray, timestamps: pd.DatetimeIndex
) -> np.ndarray:
    """Remove each station's four-year mean for each month and dataset-clock hour."""
    anomalies = np.empty_like(values, dtype=np.float64)
    groups = timestamps.month.to_numpy() * 100 + timestamps.hour.to_numpy()
    for group in np.unique(groups):
        mask = groups == group
        block = np.asarray(values[mask], dtype=np.float64)
        anomalies[mask] = block - block.mean(axis=0, keepdims=True)
    return anomalies


def window_mean(values: np.ndarray, start: int, length: int, windows: int) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)],
        axis=0,
    )
    return (cumulative[start + length:start + length + windows]
            - cumulative[start:start + windows]) / length


def core_weather(raw: np.ndarray) -> dict[str, np.ndarray]:
    get = lambda name: np.asarray(raw[..., RAW_FEATURES.index(name)], dtype=np.float64)
    u = get("100m_u_wind")
    v = get("100m_v_wind")
    return {
        "temperature_c": get("2m_temperature") - 273.15,
        "pressure_hpa": get("surface_pressure") / 100.0,
        "rh950_percent": get("relative_humidity_950"),
        "wind_speed_ms": np.hypot(u, v),
        "wind_u_ms": u,
        "wind_v_ms": v,
    }


def persistence_threshold(hours: np.ndarray, correlations: np.ndarray, level: float) -> int | None:
    below = np.flatnonzero(correlations < level)
    return int(hours[below[0]]) if len(below) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/knowair_weather_eda")
    args = parser.parse_args()

    root = Path(args.root)
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw = np.load(root / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    if raw.ndim != 3 or raw.shape[1:] != (184, 18):
        raise ValueError(f"Expected KnowAir [T, 184, 18], got {raw.shape}")
    timestamps = pd.date_range("2015-01-01", periods=len(raw), freq="3h")
    history_steps = future_steps = 24
    windows = len(raw) - history_steps - future_steps + 1
    past_hours = np.arange(-69, 1, 3)
    future_hours = np.arange(3, 73, 3)
    lead_hours = np.arange(3, 73, 3)

    weather = core_weather(raw)
    anomalies = {
        name: remove_station_month_hour_climatology(values, timestamps)
        for name, values in weather.items()
    }

    # Same-variable offset-pair relationships. With an exhaustive sliding scan,
    # each matrix cell is an autocorrelation at (future hour - past hour).
    offset_rows: list[dict[str, object]] = []
    lag_cache: dict[tuple[str, str, int], dict[str, float]] = {}
    for feature in weather:
        for representation, values in (("raw", weather[feature]), ("anomaly", anomalies[feature])):
            for lag_steps in range(1, 48):
                stats = summarize_correlations(
                    station_correlations(values[:-lag_steps], values[lag_steps:])
                )
                lag_cache[(feature, representation, lag_steps)] = stats
        for past_hour in past_hours:
            for future_hour in future_hours:
                lag_steps = (future_hour - past_hour) // 3
                row: dict[str, object] = {
                    "feature": feature,
                    "past_hour": int(past_hour),
                    "future_hour": int(future_hour),
                    "separation_hours": int(future_hour - past_hour),
                }
                for representation in ("raw", "anomaly"):
                    stats = lag_cache[(feature, representation, lag_steps)]
                    for statistic, value in stats.items():
                        row[f"{representation}_{statistic}_station_r"] = value
                offset_rows.append(row)
    offset_frame = pd.DataFrame(offset_rows)
    offset_frame.to_csv(output / "offset_pair_correlations.csv", index=False)

    # Lead-wise relation to the last observation and to the complete past mean.
    lead_rows: list[dict[str, object]] = []
    window_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for feature in weather:
        for representation, values in (("raw", weather[feature]), ("anomaly", anomalies[feature])):
            past_mean = window_mean(values, 0, history_steps, windows)
            future_mean = window_mean(values, history_steps, future_steps, windows)
            window_cache[(feature, representation)] = (past_mean, future_mean)
            last = values[history_steps - 1:history_steps - 1 + windows]
            for lead_step, lead_hour in enumerate(lead_hours, start=1):
                future = values[history_steps - 1 + lead_step:history_steps - 1 + lead_step + windows]
                for predictor, predictor_values in (("last_value", last), ("past_72h_mean", past_mean)):
                    stats = summarize_correlations(station_correlations(predictor_values, future))
                    lead_rows.append({
                        "feature": feature, "representation": representation,
                        "predictor": predictor, "lead_hour": int(lead_hour), **stats,
                    })
    lead_frame = pd.DataFrame(lead_rows)
    lead_frame.to_csv(output / "lead_correlations.csv", index=False)

    # Cross-feature pairwise relation between the past-window and future-window means.
    cross_rows: list[dict[str, object]] = []
    for representation in ("raw", "anomaly"):
        for past_feature in weather:
            past_mean = window_cache[(past_feature, representation)][0]
            for future_feature in weather:
                future_mean = window_cache[(future_feature, representation)][1]
                stats = summarize_correlations(station_correlations(past_mean, future_mean))
                cross_rows.append({
                    "representation": representation, "past_feature": past_feature,
                    "future_feature": future_feature, **stats,
                })
    cross_frame = pd.DataFrame(cross_rows)
    cross_frame.to_csv(output / "cross_feature_window_correlations.csv", index=False)

    # Circular wind-direction persistence, reported separately because Pearson r
    # on angles is invalid at the 0/360-degree wraparound.
    u, v = weather["wind_u_ms"], weather["wind_v_ms"]
    direction_rows = []
    for lead_step, lead_hour in enumerate(lead_hours, start=1):
        norm0 = np.maximum(np.hypot(u[:-lead_step], v[:-lead_step]), 1e-12)
        norm1 = np.maximum(np.hypot(u[lead_step:], v[lead_step:]), 1e-12)
        cosine = (u[:-lead_step] * u[lead_step:] + v[:-lead_step] * v[lead_step:]) / (norm0 * norm1)
        station_similarity = np.clip(cosine, -1, 1).mean(axis=0)
        direction_rows.append({
            "lead_hour": int(lead_hour),
            "median_station_mean_cosine_similarity": float(np.median(station_similarity)),
            "median_station_mean_angular_difference_deg": float(
                np.median(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean(axis=0))
            ),
        })
    direction_frame = pd.DataFrame(direction_rows)
    direction_frame.to_csv(output / "wind_direction_persistence.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    plot_features = list(weather)
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    for axis, feature in zip(axes.flat, plot_features):
        subset = offset_frame[offset_frame.feature == feature]
        matrix = subset.pivot(index="past_hour", columns="future_hour", values="anomaly_median_station_r")
        sns.heatmap(matrix, ax=axis, cmap="vlag", center=0, vmin=-1, vmax=1,
                    xticklabels=4, yticklabels=4, cbar_kws={"label": "median station r"})
        axis.set_title(FEATURE_LABELS[feature])
        axis.set_xlabel("Future hour relative to origin")
        axis.set_ylabel("Past hour relative to origin")
    fig.suptitle("KnowAir 72h past x 72h future: de-seasonalized correlation", fontsize=16)
    fig.savefig(output / "offset_pair_anomaly_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, feature in zip(axes.flat, plot_features):
        subset = lead_frame[(lead_frame.feature == feature) & (lead_frame.representation == "anomaly")]
        for predictor, style in (("last_value", "-o"), ("past_72h_mean", "--s")):
            line = subset[subset.predictor == predictor]
            axis.plot(line.lead_hour, line["median"], style, ms=3, label=predictor.replace("_", " "))
            axis.fill_between(line.lead_hour, line.q25, line.q75, alpha=0.12)
        axis.axhline(0, color="black", lw=.7)
        axis.set(title=FEATURE_LABELS[feature], xlabel="Forecast lead (hours)", ylabel="median station r", ylim=(-.35, 1))
        axis.legend(fontsize=8)
    fig.suptitle("Information retained from the past after removing station-month-hour climatology", fontsize=15)
    fig.savefig(output / "lead_correlation_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, representation in zip(axes, ("raw", "anomaly")):
        subset = cross_frame[cross_frame.representation == representation]
        matrix = subset.pivot(index="past_feature", columns="future_feature", values="median")
        matrix = matrix.loc[plot_features, plot_features]
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                    ax=axis, xticklabels=[FEATURE_LABELS[x] for x in plot_features],
                    yticklabels=[FEATURE_LABELS[x] for x in plot_features])
        axis.set_title(f"{representation.capitalize()} 72h-window means")
        axis.set_xlabel("Future feature")
        axis.set_ylabel("Past feature")
    fig.savefig(output / "cross_feature_correlations.png", dpi=180)
    plt.close(fig)

    key_features = ("temperature_c", "pressure_hpa", "rh950_percent", "wind_speed_ms")
    summary: dict[str, object] = {
        "dataset": {
            "time_start": str(timestamps[0]), "time_end": str(timestamps[-1]),
            "cadence_hours": 3, "time_steps": len(raw), "stations": raw.shape[1],
            "valid_overlapping_windows": windows,
            "window_definition": "24 past steps (72h) followed by 24 future steps (72h)",
            "split_boundaries_used": False,
        },
        "method": {
            "origin_selection": "exhaustive sliding scan; windows overlap",
            "aggregation": "Pearson r per station, then median and IQR across 184 stations",
            "anomaly": "station x calendar-month x dataset-clock-hour climatological mean removed",
        },
        "anomaly_last_value_correlation": {},
        "anomaly_past_mean_to_future_mean": {},
        "wind_direction": direction_rows,
    }
    for feature in key_features:
        curve = lead_frame[
            (lead_frame.feature == feature) & (lead_frame.representation == "anomaly")
            & (lead_frame.predictor == "last_value")
        ].sort_values("lead_hour")
        correlations = curve["median"].to_numpy()
        summary["anomaly_last_value_correlation"][feature] = {
            "by_lead_hour": {str(int(h)): float(r) for h, r in zip(curve.lead_hour, correlations)},
            "first_lead_below_r_0.5_hours": persistence_threshold(curve.lead_hour.to_numpy(), correlations, .5),
            "first_lead_below_r_0.3_hours": persistence_threshold(curve.lead_hour.to_numpy(), correlations, .3),
        }
        row = cross_frame[
            (cross_frame.representation == "anomaly")
            & (cross_frame.past_feature == feature) & (cross_frame.future_feature == feature)
        ].iloc[0]
        summary["anomaly_past_mean_to_future_mean"][feature] = {
            key: float(row[key]) for key in ("median", "q25", "q75")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
