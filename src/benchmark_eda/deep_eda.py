"""Deep, leakage-aware diagnostics for the UCI Beijing multi-site dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .data import (
    POLLUTANT_FEATURES,
    WEATHER_FEATURES,
    WIND_ANGLE,
    _haversine_graph,
    load_raw_frames,
)


HORIZONS = (1, 6, 12, 24)
SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def _full_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    lon = np.radians(coordinates[:, 0])
    lat = np.radians(coordinates[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _season(month: pd.Series) -> pd.Series:
    return month.map(SEASON).astype("category")


def _run_lengths(flag: np.ndarray) -> list[int]:
    padded = np.r_[False, flag, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return (ends - starts).tolist()


def _load_context(raw_dir: Path, coords_path: Path):
    frame = load_raw_frames(raw_dir).copy()
    frame = frame.sort_values(["station", "timestamp"]).reset_index(drop=True)
    angle = frame["wd"].map(WIND_ANGLE) * np.pi / 180
    frame["WD_SIN"] = np.sin(angle)
    frame["WD_COS"] = np.cos(angle)
    frame["FLOW_EAST"] = -frame["WD_SIN"]
    frame["FLOW_NORTH"] = -frame["WD_COS"]
    frame["season"] = _season(frame.timestamp.dt.month)

    stations = sorted(frame.station.unique())
    timestamps = pd.DatetimeIndex(sorted(frame.timestamp.unique()))
    train_end = timestamps[int(len(timestamps) * .70)]
    val_end = timestamps[int(len(timestamps) * .85)]
    frame["split"] = np.select(
        [frame.timestamp < train_end, frame.timestamp < val_end],
        ["train", "val"], default="test",
    )
    coords_frame = pd.read_csv(coords_path).set_index("station")
    coordinates = coords_frame.loc[
        stations, ["longitude", "latitude"]
    ].to_numpy(np.float32)
    neighbor_index, neighbor_distance, neighbor_direction = _haversine_graph(
        coordinates, 4
    )
    wide_pm = frame.pivot(
        index="timestamp", columns="station", values="PM2.5"
    ).reindex(columns=stations)
    return {
        "frame": frame,
        "stations": stations,
        "timestamps": timestamps,
        "train_end": train_end,
        "val_end": val_end,
        "coordinates": coordinates,
        "neighbor_index": neighbor_index,
        "neighbor_distance": neighbor_distance,
        "neighbor_direction": neighbor_direction,
        "wide_pm": wide_pm,
    }


def regional_and_functional_graph(context, output_dir: Path) -> dict:
    wide = context["wide_pm"]
    stations = context["stations"]
    city = wide.mean(axis=1)
    residual = wide.sub(city, axis=0)
    change = wide.diff()
    residual_change = residual.diff()
    distance = _full_distance_matrix(context["coordinates"])

    regional_rows = []
    for horizon in HORIZONS:
        total_delta = wide.shift(-horizon) - wide
        regional_delta = city.shift(-horizon) - city
        local_delta = total_delta.sub(regional_delta, axis=0)
        regional_var = float(np.nanvar(regional_delta))
        local_var = float(np.nanvar(local_delta.to_numpy()))
        total_var = float(np.nanvar(total_delta.to_numpy()))
        regional_rows.append(
            {
                "horizon": horizon,
                "total_change_variance": total_var,
                "regional_change_variance": regional_var,
                "local_change_variance": local_var,
                "regional_share_of_component_variance": regional_var
                / (regional_var + local_var),
            }
        )
    regional = pd.DataFrame(regional_rows)
    regional.to_csv(output_dir / "regional_local_variance.csv", index=False)

    pair_rows = []
    for i, target in enumerate(stations):
        for j in range(i + 1, len(stations)):
            source = stations[j]
            pair_rows.append(
                {
                    "station_a": target,
                    "station_b": source,
                    "distance_km": distance[i, j],
                    "raw_level_rho": wide[target].corr(
                        wide[source], method="spearman"
                    ),
                    "change_rho": change[target].corr(
                        change[source], method="spearman"
                    ),
                    "regional_residual_rho": residual[target].corr(
                        residual[source], method="spearman"
                    ),
                    "residual_change_rho": residual_change[target].corr(
                        residual_change[source], method="spearman"
                    ),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output_dir / "pairwise_functional_graph.csv", index=False)

    residual_corr = residual_change.corr(method="spearman")
    overlaps = []
    for target_idx, target in enumerate(stations):
        empirical = (
            residual_corr[target].drop(target).sort_values(ascending=False).head(4)
        )
        geographic = {stations[idx] for idx in context["neighbor_index"][target_idx]}
        overlaps.append(len(set(empirical.index) & geographic) / 4)

    return {
        "regional_share": regional.set_index("horizon")[
            "regional_share_of_component_variance"
        ].to_dict(),
        "distance_vs_raw_level_rho": spearmanr(
            pairs.distance_km, pairs.raw_level_rho, nan_policy="omit"
        ).statistic,
        "distance_vs_residual_change_rho": spearmanr(
            pairs.distance_km, pairs.residual_change_rho, nan_policy="omit"
        ).statistic,
        "geographic_knn_overlap_with_functional_top4_mean": float(np.mean(overlaps)),
        "geographic_knn_overlap_by_station": dict(zip(stations, overlaps)),
    }


def spike_coherence_and_missingness(context, output_dir: Path) -> dict:
    wide = context["wide_pm"]
    train = wide.loc[wide.index < context["train_end"]]
    thresholds = train.quantile(.90)
    observed = wide.notna()
    exceed = wide.ge(thresholds, axis=1) & observed
    simultaneous = exceed.sum(axis=1)
    active = simultaneous > 0
    active_counts = simultaneous.loc[active]

    spike_categories = pd.cut(
        active_counts,
        bins=[0, 2, 5, 12],
        labels=["isolated_1_2", "cluster_3_5", "regional_6_12"],
        include_lowest=True,
    )
    category_share = spike_categories.value_counts(normalize=True).sort_index()

    episode_rows = []
    padded = np.r_[False, active.to_numpy(), False].astype(np.int8)
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    for start, end in zip(starts, ends):
        counts = simultaneous.iloc[start:end].to_numpy()
        episode_rows.append(
            {
                "start": wide.index[start],
                "duration_h": end - start,
                "peak_simultaneous_stations": int(counts.max()),
                "hours_to_peak": int(np.argmax(counts)),
                "reached_half_network": bool((counts >= 6).any()),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(output_dir / "spike_episodes.csv", index=False)

    split = pd.Series(
        np.select(
            [wide.index < context["train_end"], wide.index < context["val_end"]],
            ["train", "val"], default="test",
        ),
        index=wide.index,
    )
    season = pd.Series(_season(wide.index.to_series().dt.month).to_numpy(), index=wide.index)
    exceedance_by_regime = pd.DataFrame(
        {
            "split": split,
            "season": season,
            "simultaneous": simultaneous,
            "any_spike": active,
            "regional_spike": simultaneous >= 6,
        }
    )
    spike_regime = (
        exceedance_by_regime.groupby(["split", "season"], observed=True)
        .agg(
            hours=("any_spike", "size"),
            any_spike_rate=("any_spike", "mean"),
            regional_spike_rate=("regional_spike", "mean"),
            mean_simultaneous=("simultaneous", "mean"),
        )
        .reset_index()
    )
    spike_regime.to_csv(output_dir / "spike_regime.csv", index=False)

    missing_count = wide.isna().sum(axis=1)
    city_pm = wide.mean(axis=1)
    city_bin = pd.qcut(city_pm, 4, labels=["q1", "q2", "q3", "q4"])
    missing_table = pd.DataFrame(
        {
            "missing_stations": missing_count,
            "city_pm_bin": city_bin,
            "split": split,
            "season": season,
        }
    )
    missing_by_regime = (
        missing_table.groupby(["split", "season", "city_pm_bin"], observed=True)
        .missing_stations.agg(["count", "mean", "max"])
        .reset_index()
    )
    missing_by_regime.to_csv(output_dir / "missingness_regime.csv", index=False)
    gap_lengths = []
    for station in context["stations"]:
        gap_lengths.extend(_run_lengths(wide[station].isna().to_numpy()))

    return {
        "train_station_p90": thresholds.to_dict(),
        "active_spike_hour_category_share": category_share.to_dict(),
        "episode_count": len(episodes),
        "episode_duration_quantiles_h": episodes.duration_h.quantile(
            [.5, .75, .9, .95]
        ).to_dict(),
        "episodes_reaching_half_network_rate": episodes.reached_half_network.mean(),
        "median_hours_to_peak": episodes.hours_to_peak.median(),
        "timestamps_with_3plus_missing_stations_rate": float((missing_count >= 3).mean()),
        "missing_gap_quantiles_h": pd.Series(gap_lengths).quantile(
            [.5, .9, .99, 1]
        ).to_dict(),
    }


def wind_conditioned_lead_lag(context, output_dir: Path) -> dict:
    frame = context["frame"]
    stations = context["stations"]
    wide = context["wide_pm"]
    changes = wide.diff()
    speed = frame.pivot(index="timestamp", columns="station", values="WSPM")
    flow_east = frame.pivot(index="timestamp", columns="station", values="FLOW_EAST")
    flow_north = frame.pivot(index="timestamp", columns="station", values="FLOW_NORTH")

    rows = []
    best_rows = []
    robustness_rows = []
    lags = (0, 1, 3, 6, 12)
    split = pd.Series(
        np.select(
            [wide.index < context["train_end"], wide.index < context["val_end"]],
            ["train", "val"], default="test",
        ),
        index=wide.index,
    )
    season = pd.Series(
        _season(wide.index.to_series().dt.month).to_numpy(), index=wide.index
    )
    for target_idx, target in enumerate(stations):
        for edge_idx, source_idx in enumerate(context["neighbor_index"][target_idx]):
            source = stations[source_idx]
            direction = context["neighbor_direction"][target_idx, edge_idx]
            alignment = flow_east[source] * direction[0] + flow_north[source] * direction[1]
            source_speed = speed[source]
            regime = pd.Series("cross", index=wide.index, dtype="object")
            regime.loc[source_speed < 1] = "calm"
            regime.loc[(source_speed >= 1) & (alignment >= .5)] = "aligned"
            regime.loc[(source_speed >= 1) & (alignment <= -.5)] = "opposed"
            source_change = changes[source]
            correlations = {}
            for lag in lags:
                target_future_change = changes[target].shift(-lag)
                correlations[lag] = source_change.corr(
                    target_future_change, method="spearman"
                )
                for name in ("calm", "cross", "aligned", "opposed"):
                    mask = regime.eq(name)
                    pair = pd.concat(
                        [source_change[mask], target_future_change[mask]], axis=1
                    ).dropna()
                    rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "distance_km": context["neighbor_distance"][target_idx, edge_idx],
                            "lag_h": lag,
                            "wind_regime": name,
                            "n": len(pair),
                            "rho": rho,
                        }
                    )

            # The lag-one contrast is the only directional effect suggested by
            # the aggregate screen.  Re-estimate it within time and weather
            # strata so a single season/split cannot manufacture the result.
            target_lag1 = changes[target].shift(-1)
            strata = {
                **{f"split:{value}": split.eq(value) for value in ("train", "val", "test")},
                **{
                    f"season:{value}": season.eq(value)
                    for value in ("winter", "spring", "summer", "autumn")
                },
                "speed:moderate": source_speed.ge(1) & source_speed.lt(3),
                "speed:strong": source_speed.ge(3),
            }
            for stratum, stratum_mask in strata.items():
                result = {
                    "source": source,
                    "target": target,
                    "stratum": stratum,
                }
                for name in ("aligned", "opposed"):
                    mask = stratum_mask & regime.eq(name)
                    pair = pd.concat(
                        [source_change[mask], target_lag1[mask]], axis=1
                    ).dropna()
                    result[f"{name}_n"] = len(pair)
                    result[f"{name}_rho"] = pair.iloc[:, 0].corr(
                        pair.iloc[:, 1], method="spearman"
                    )
                result["aligned_minus_opposed"] = (
                    result["aligned_rho"] - result["opposed_rho"]
                )
                robustness_rows.append(result)
            best_lag = max(correlations, key=lambda lag: abs(correlations[lag]))
            best_rows.append(
                {
                    "source": source,
                    "target": target,
                    "distance_km": context["neighbor_distance"][target_idx, edge_idx],
                    "best_lag_h": best_lag,
                    "best_rho": correlations[best_lag],
                }
            )
    lag_table = pd.DataFrame(rows)
    lag_table.to_csv(output_dir / "wind_conditioned_pair_lag.csv", index=False)
    best = pd.DataFrame(best_rows)
    best.to_csv(output_dir / "pair_best_lag.csv", index=False)
    aggregate = (
        lag_table.groupby(["lag_h", "wind_regime"])
        .agg(pair_count=("rho", "count"), median_rho=("rho", "median"), mean_rho=("rho", "mean"))
        .reset_index()
    )
    aggregate.to_csv(output_dir / "wind_lag_summary.csv", index=False)

    pivot = lag_table.pivot_table(
        index=["source", "target", "lag_h"], columns="wind_regime", values="rho"
    ).reset_index()
    paired = pivot.dropna(subset=["aligned", "opposed"]).copy()
    paired["aligned_minus_opposed"] = paired.aligned - paired.opposed
    paired_summary = paired.groupby("lag_h").aligned_minus_opposed.agg(
        ["count", "mean", "median", "std"]
    )
    paired_summary["positive_pair_fraction"] = paired.groupby(
        "lag_h"
    ).aligned_minus_opposed.apply(lambda values: (values > 0).mean())
    paired_summary.to_csv(output_dir / "wind_alignment_contrast.csv")

    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(output_dir / "wind_alignment_robustness_pairs.csv", index=False)
    robust_summary = (
        robustness.dropna(subset=["aligned_minus_opposed"])
        .groupby("stratum")
        .aligned_minus_opposed.agg(
            pair_count="count", mean="mean", median="median", std="std",
            positive_pair_fraction=lambda values: (values > 0).mean(),
        )
        .reset_index()
    )
    robust_summary.to_csv(output_dir / "wind_alignment_robustness.csv", index=False)

    return {
        "best_lag_distribution": best.best_lag_h.value_counts(normalize=True).to_dict(),
        "median_best_lag_h": float(best.best_lag_h.median()),
        "aligned_minus_opposed_by_lag": paired_summary.to_dict(orient="index"),
        "lag1_alignment_robustness": robust_summary.set_index("stratum").to_dict(
            orient="index"
        ),
        "pair_count": len(best),
    }


def _engineered_frame(context) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = context["frame"].copy()
    group = frame.groupby("station", sort=False)
    for lag in (1, 3, 6, 24):
        frame[f"PM_trend_{lag}h"] = frame["PM2.5"] - group["PM2.5"].shift(lag)
    for feature in WEATHER_FEATURES:
        if feature not in frame:
            continue
        frame[f"{feature}_mean24"] = group[feature].transform(
            lambda values: values.rolling(24, min_periods=6).mean()
        )
    hour = frame.timestamp.dt.hour
    day = frame.timestamp.dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["annual_cos"] = np.cos(2 * np.pi * day / 365.25)

    spatial_parts = []
    wide = context["wide_pm"]
    for target_idx, station in enumerate(context["stations"]):
        neighbours = [
            context["stations"][idx]
            for idx in context["neighbor_index"][target_idx]
        ]
        neighbour_panel = wide[neighbours]
        neighbour_mean = neighbour_panel.mean(axis=1)
        spatial_parts.append(
            pd.DataFrame(
                {
                    "timestamp": wide.index,
                    "station": station,
                    "neighbor_level": neighbour_mean,
                    "neighbor_trend_1h": neighbour_mean.diff(),
                    "neighbor_trend_6h": neighbour_mean.diff(6),
                    "neighbor_dispersion": neighbour_panel.std(axis=1),
                }
            )
        )
    spatial = pd.concat(spatial_parts, ignore_index=True)
    frame = frame.merge(spatial, on=["timestamp", "station"], how="left")
    frame["spatial_gap"] = frame.neighbor_level - frame["PM2.5"]

    local = list(POLLUTANT_FEATURES) + [
        "PM_trend_1h", "PM_trend_3h", "PM_trend_6h", "PM_trend_24h",
        "hour_sin", "hour_cos", "annual_sin", "annual_cos",
    ]
    meteo = [
        feature for feature in WEATHER_FEATURES if feature in frame
    ] + [
        f"{feature}_mean24"
        for feature in WEATHER_FEATURES
        if f"{feature}_mean24" in frame
    ]
    spatial_features = [
        "neighbor_level", "neighbor_trend_1h", "neighbor_trend_6h",
        "neighbor_dispersion", "spatial_gap",
    ]
    station_dummy = pd.get_dummies(frame.station, prefix="station", dtype=np.float32)
    frame = pd.concat([frame, station_dummy], axis=1)
    local += station_dummy.columns.tolist()
    return frame, {"local": local, "meteo": meteo, "spatial": spatial_features}


def _ridge_fit_predict(
    frame: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = frame.loc[train_mask, features].astype(np.float32)
    test_x = frame.loc[test_mask, features].astype(np.float32)
    median = train_x.median()
    train_x = train_x.fillna(median).fillna(0)
    test_x = test_x.fillna(median).fillna(0)
    mean = train_x.mean()
    std = train_x.std().replace(0, 1).fillna(1)
    train_array = ((train_x - mean) / std).to_numpy(np.float32)
    test_array = ((test_x - mean) / std).to_numpy(np.float32)
    model = Ridge(alpha=10.0)
    model.fit(train_array, target.loc[train_mask].to_numpy())
    return model.predict(train_array), model.predict(test_array)


def conditional_incremental_value(context, output_dir: Path) -> dict:
    frame, feature_sets = _engineered_frame(context)
    variants = {
        "local": feature_sets["local"],
        "local_meteo": feature_sets["local"] + feature_sets["meteo"],
        "local_spatial": feature_sets["local"] + feature_sets["spatial"],
        "full": feature_sets["local"] + feature_sets["meteo"] + feature_sets["spatial"],
    }
    group = frame.groupby("station", sort=False)
    train_rows = frame.timestamp < context["train_end"]
    test_rows = frame.timestamp >= context["val_end"]
    station_p50 = frame.loc[train_rows].groupby("station")["PM2.5"].quantile(.50)
    station_p90 = frame.loc[train_rows].groupby("station")["PM2.5"].quantile(.90)
    current_p50 = frame.station.map(station_p50)
    current_p90 = frame.station.map(station_p90)
    frame["pollution_regime"] = np.select(
        [frame["PM2.5"] < current_p50, frame["PM2.5"] < current_p90],
        ["low", "middle"], default="high",
    )
    frame["wind_regime"] = pd.cut(
        frame.WSPM, [-np.inf, 1, 3, np.inf], labels=["calm", "moderate", "strong"]
    )
    gap_cut = frame.loc[train_rows, "spatial_gap"].abs().quantile([.33, .67])
    frame["gap_magnitude"] = pd.cut(
        frame.spatial_gap.abs(),
        [-np.inf, gap_cut.iloc[0], gap_cut.iloc[1], np.inf],
        labels=["small", "medium", "large"],
    )
    signed_cut = frame.loc[train_rows, "spatial_gap"].abs().quantile(.67)
    frame["gap_direction"] = np.select(
        [frame.spatial_gap < -signed_cut, frame.spatial_gap > signed_cut],
        ["neighbors_lower", "neighbors_higher"], default="similar",
    )

    metric_rows = []
    gain_rows = []
    horizon_summary = {}
    for horizon in HORIZONS:
        target = group["PM2.5"].shift(-horizon) - frame["PM2.5"]
        target_time = frame.timestamp + pd.to_timedelta(horizon, unit="h")
        train_mask = (
            train_rows & target.notna() & (target_time < context["train_end"])
        ).to_numpy()
        test_mask = (test_rows & target.notna()).to_numpy()
        predictions = {}
        for variant, features in variants.items():
            _, prediction = _ridge_fit_predict(
                frame, features, target, train_mask, test_mask
            )
            predictions[variant] = prediction
            error = prediction - target.loc[test_mask].to_numpy()
            metric_rows.append(
                {
                    "horizon": horizon,
                    "variant": variant,
                    "n": len(error),
                    "mae": np.mean(np.abs(error)),
                    "rmse": np.sqrt(np.mean(error**2)),
                }
            )
        baseline_error = np.abs(
            predictions["local_meteo"] - target.loc[test_mask].to_numpy()
        )
        full_error = np.abs(
            predictions["full"] - target.loc[test_mask].to_numpy()
        )
        diagnostic = frame.loc[test_mask, [
            "station", "season", "pollution_regime", "wind_regime",
            "gap_magnitude", "gap_direction",
        ]].copy()
        diagnostic["spatial_gain_after_meteo"] = baseline_error - full_error
        for grouping in (
            "station", "season", "pollution_regime", "wind_regime",
            "gap_magnitude", "gap_direction",
        ):
            grouped = diagnostic.groupby(grouping, observed=True)[
                "spatial_gain_after_meteo"
            ]
            for value, values in grouped:
                gain_rows.append(
                    {
                        "horizon": horizon,
                        "grouping": grouping,
                        "group": str(value),
                        "n": len(values),
                        "mean_gain": values.mean(),
                        "median_gain": values.median(),
                        "fraction_helped": (values > 0).mean(),
                    }
                )
        horizon_summary[horizon] = {
            "spatial_gain_after_meteo_mean": float(
                (baseline_error - full_error).mean()
            )
        }
    metrics = pd.DataFrame(metric_rows)
    gains = pd.DataFrame(gain_rows)
    metrics.to_csv(output_dir / "diagnostic_ridge_ablation.csv", index=False)
    gains.to_csv(output_dir / "spatial_gain_by_regime.csv", index=False)
    return {
        "ridge_alpha": 10.0,
        "horizon_summary": horizon_summary,
        "largest_spatial_gain_groups": gains.sort_values(
            "mean_gain", ascending=False
        ).head(12).to_dict(orient="records"),
        "smallest_spatial_gain_groups": gains.sort_values(
            "mean_gain"
        ).head(12).to_dict(orient="records"),
    }


def _make_figures(output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    regional = pd.read_csv(output_dir / "regional_local_variance.csv")
    ridge = pd.read_csv(output_dir / "diagnostic_ridge_ablation.csv")
    wind = pd.read_csv(output_dir / "wind_lag_summary.csv")
    gains = pd.read_csv(output_dir / "spatial_gain_by_regime.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(
        regional.horizon,
        regional.regional_share_of_component_variance,
        marker="o",
    )
    axes[0, 0].set(
        title="Regional share of future-change component variance",
        xlabel="Horizon (h)", ylabel="Share",
    )
    sns.lineplot(data=ridge, x="horizon", y="mae", hue="variant", marker="o", ax=axes[0, 1])
    axes[0, 1].set_title("Leakage-safe diagnostic Ridge MAE")
    sns.lineplot(data=wind, x="lag_h", y="median_rho", hue="wind_regime", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("Pair change correlation by wind alignment")
    selected = gains.query(
        "horizon == 24 and grouping in ['gap_direction', 'wind_regime', 'pollution_regime']"
    )
    sns.barplot(data=selected, x="group", y="mean_gain", hue="grouping", ax=axes[1, 1])
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].set_title("Spatial gain after meteo at h=24")
    plt.tight_layout()
    fig.savefig(output_dir / "deep_eda_main.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_deep_eda(
    raw_dir: str | Path = "data/raw/PRSA_Data_20130301-20170228",
    coords_path: str | Path = "data/metadata/uci_beijing_station_coords.csv",
    output_dir: str | Path = "artifacts/deep_eda_uci",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(Path(raw_dir), Path(coords_path))
    summary = {
        "scope": {
            "dataset": "UCI Beijing Multi-Site Air Quality",
            "stations": len(context["stations"]),
            "start": context["timestamps"].min(),
            "end": context["timestamps"].max(),
            "train_end": context["train_end"],
            "val_end": context["val_end"],
            "note": "Diagnostic associations are hypothesis-generating, not causal claims.",
        },
        "regional_functional_graph": regional_and_functional_graph(context, output_dir),
        "spike_missingness": spike_coherence_and_missingness(context, output_dir),
        "wind_pair_lag": wind_conditioned_lead_lag(context, output_dir),
        "conditional_incremental_value": conditional_incremental_value(context, output_dir),
    }
    _make_figures(output_dir)
    (output_dir / "deep_eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary
