"""Causal and oracle routing among low/center/high PM trajectories on KnowAir.

The center expert is the protected Boundary-CSIO forecast; the low and high
experts are the frozen DART 10th/90th percentile trajectories.  A router uses
only quantities available at forecast origin: historical PM summaries, DART
uncertainty/regime probabilities and cross-fitted weather-event arrival
probabilities.  Router architecture and fallback are chosen on the chronological
tail of train, then evaluated once on validation.  Test is never indexed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from shift_pm.data import KnowAirDataModule


RULES = ("hard", "soft")
CONFIDENCE_THRESHOLDS = (0.0, 0.40, 0.50, 0.60)


def expert_tensor(quantiles: np.ndarray, center: np.ndarray) -> np.ndarray:
    quantiles = quantiles.astype(np.float32)
    return np.stack((quantiles[:, 0], center, quantiles[:, 2]), axis=1)


def expert_labels(
    experts: np.ndarray, target: np.ndarray, observed: np.ndarray, day: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    part = slice(day * 8, (day + 1) * 8)
    mask = observed[:, part]
    count = mask.sum(axis=1)
    error = np.where(
        mask[:, None], np.abs(experts[:, :, part] - target[:, None, part]), 0.0
    ).sum(axis=2) / np.maximum(count[:, None], 1)
    ordered = np.sort(error, axis=1)
    label = np.argmin(error, axis=1).astype(np.int8)
    margin = ordered[:, 1] - ordered[:, 0]
    return label, margin, count > 0


def _summaries(values: np.ndarray) -> list[np.ndarray]:
    return [
        values.mean(axis=1),
        values.std(axis=1),
        values[:, 0],
        values[:, -1],
        values[:, -1] - values[:, 0],
    ]


def router_features(
    data: KnowAirDataModule,
    origins: np.ndarray,
    experts: np.ndarray,
    prediction_std: np.ndarray,
    high_probability: np.ndarray,
    event_probability: np.ndarray,
    day: int,
) -> np.ndarray:
    """Return [origin, station, feature] causal router features."""
    part = slice(day * 8, (day + 1) * 8)
    pieces: list[np.ndarray] = []
    for expert in range(3):
        pieces.extend(_summaries(experts[:, expert, part]))
    pieces.extend(_summaries(prediction_std[:, part]))
    pieces.extend(_summaries(high_probability[:, part]))
    pieces.extend((
        (experts[:, 2, part] - experts[:, 0, part]).mean(axis=1),
        (experts[:, 1, part] - experts[:, 0, part]).mean(axis=1),
        (experts[:, 2, part] - experts[:, 1, part]).mean(axis=1),
    ))

    local_event = event_probability.transpose(0, 3, 1, 2).reshape(
        len(origins), data.stations, -1
    )
    regional_event = event_probability.mean(axis=-1).reshape(len(origins), -1)
    regional_event = np.broadcast_to(
        regional_event[:, None],
        (len(origins), data.stations, regional_event.shape[1]),
    )
    entropy = -np.sum(
        event_probability * np.log(np.maximum(event_probability, 1e-6)), axis=2
    ).transpose(0, 2, 1)

    pm_parts = []
    latest = np.empty((len(origins), data.stations), dtype=np.float32)
    regional_latest = np.empty(len(origins), dtype=np.float32)
    regional_recent = np.empty(len(origins), dtype=np.float32)
    regional_tendency = np.empty(len(origins), dtype=np.float32)
    rank = np.empty((len(origins), data.stations), dtype=np.float32)
    for index, origin in enumerate(origins):
        history = data.pm_anomaly[origin - 24:origin].astype(np.float32)
        latest[index] = history[-1]
        regional_series = history.mean(axis=1)
        regional_latest[index] = regional_series[-1]
        regional_recent[index] = regional_series[-8:].mean()
        regional_tendency[index] = regional_series[-1] - regional_series[-8]
        order = np.argsort(np.argsort(history[-1], kind="mergesort"), kind="mergesort")
        rank[index] = order / max(data.stations - 1, 1)
        for width in (2, 4, 8, 16, 24):
            if len(pm_parts) < 10:
                pm_parts.extend((
                    np.empty((len(origins), data.stations), dtype=np.float32),
                    np.empty((len(origins), data.stations), dtype=np.float32),
                ))
            pm_parts[2 * (2, 4, 8, 16, 24).index(width)][index] = history[-width:].mean(0)
            pm_parts[2 * (2, 4, 8, 16, 24).index(width) + 1][index] = history[-width:].std(0)

    regional = np.stack((regional_latest, regional_recent, regional_tendency), axis=1)
    regional = np.broadcast_to(
        regional[:, None], (len(origins), data.stations, regional.shape[1])
    )
    pm_feature = np.stack(
        [latest, rank, latest - regional_latest[:, None], *pm_parts], axis=2
    )
    coordinates = np.broadcast_to(
        data.coordinates[None], (len(origins), data.stations, 2)
    )
    timestamp = data.timestamps[origins]
    calendar = np.stack((
        np.sin(2 * np.pi * timestamp.dayofyear.to_numpy() / 365.25),
        np.cos(2 * np.pi * timestamp.dayofyear.to_numpy() / 365.25),
        np.sin(2 * np.pi * timestamp.hour.to_numpy() / 24.0),
        np.cos(2 * np.pi * timestamp.hour.to_numpy() / 24.0),
    ), axis=1)
    calendar = np.broadcast_to(
        calendar[:, None], (len(origins), data.stations, 4)
    )
    scalar_pieces = np.stack(pieces, axis=2)
    return np.concatenate((
        scalar_pieces,
        local_event,
        regional_event,
        entropy,
        pm_feature,
        regional,
        coordinates,
        calendar,
    ), axis=2).astype(np.float32)


def routed_prediction(
    experts: np.ndarray,
    probability: np.ndarray,
    day: int,
    rule: str,
    confidence: float,
) -> np.ndarray:
    part = slice(day * 8, (day + 1) * 8)
    if rule == "hard":
        choice = np.argmax(probability, axis=2)
        selected = np.take_along_axis(
            experts[:, :, part], choice[:, None, None, :], axis=1
        )[:, 0]
    elif rule == "soft":
        selected = np.sum(
            experts[:, :, part] * probability.transpose(0, 2, 1)[:, :, None],
            axis=1,
        )
    else:
        raise ValueError(f"unknown routing rule {rule}")
    use_router = probability.max(axis=2) >= confidence
    return np.where(use_router[:, None], selected, experts[:, 1, part])


def mae(
    prediction: np.ndarray, target: np.ndarray, observed: np.ndarray
) -> float:
    return float(np.abs(prediction - target)[observed].mean())


def oracle_selector_metrics(
    experts: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
) -> tuple[list[dict[str, object]], float]:
    """Evaluate the station-day oracle selector used for the 13.877 ceiling."""
    rows: list[dict[str, object]] = []
    weighted_error = 0.0
    total = 0
    for day in range(3):
        part = slice(day * 8, (day + 1) * 8)
        label, _, valid = expert_labels(experts, target, observed, day)
        oracle = np.take_along_axis(
            experts[:, :, part], label[:, None, None, :], axis=1
        )[:, 0]
        mask = observed[:, part]
        score = mae(oracle, target[:, part], mask)
        count = int(mask.sum())
        rows.append({
            "future_day": day + 1,
            "oracle_station_day_mae": score,
            "valid_station_days": int(valid.sum()),
            "observations": count,
        })
        weighted_error += score * count
        total += count
    return rows, weighted_error / total


def build_classifier(
    max_leaf_nodes: int, min_samples_leaf: int, seed: int
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=80,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=10.0,
        random_state=seed,
    )


def flatten_selected(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    origin_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = features[origin_indices].reshape(-1, features.shape[-1])
    y = labels[origin_indices].reshape(-1)
    selected = valid[origin_indices].reshape(-1)
    return x[selected], y[selected], selected


def purged_train_tail_indices(
    length: int,
    horizon: int,
    origin_stride: int,
    split_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Split router fitting and tuning without overlapping target windows.

    Router labels consume the full target horizon.  The last fitting origin
    must therefore finish no later than the first chronological tuning origin.
    """
    if horizon <= 0 or origin_stride <= 0:
        raise ValueError("horizon and origin_stride must be positive")
    split = max(128, int(split_fraction * length))
    if split >= length:
        raise ValueError("train-tail split leaves no tuning origins")
    train_stop = split - horizon + 1
    if train_stop <= 0:
        raise ValueError("purge gap leaves no fitting origins")
    train_origins = np.arange(0, train_stop, origin_stride, dtype=np.int64)
    query_origins = np.arange(split, length, dtype=np.int64)
    if train_origins[-1] + horizon > query_origins[0]:
        raise AssertionError("purge failure: fitting and tuning targets overlap")
    return train_origins, query_origins


def expand_probability(
    classifier: HistGradientBoostingClassifier,
    features: np.ndarray,
) -> np.ndarray:
    flat = classifier.predict_proba(features.reshape(-1, features.shape[-1]))
    probability = np.zeros((len(flat), 3), dtype=np.float64)
    probability[:, classifier.classes_.astype(int)] = flat
    return probability.reshape(len(features), features.shape[1], 3)


def fit_and_select(
    features: np.ndarray,
    labels: np.ndarray,
    margin: np.ndarray,
    valid: np.ndarray,
    experts: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    day: int,
    seed: int,
    origin_stride: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_origins, query_origins = purged_train_tail_indices(
        len(features), experts.shape[2], origin_stride
    )
    x, y, selected = flatten_selected(
        features, labels, valid, train_origins
    )
    sample_margin = margin[train_origins].reshape(-1)[selected]
    sample_weight = 1.0 + np.minimum(sample_margin / 5.0, 4.0)
    rows = []
    best = None
    best_mae = np.inf
    part = slice(day * 8, (day + 1) * 8)
    for leaves in (7, 15):
        for minimum in (100, 300):
            classifier = build_classifier(leaves, minimum, seed)
            classifier.fit(x, y, sample_weight=sample_weight)
            probability = expand_probability(classifier, features[query_origins])
            for rule in RULES:
                for confidence in CONFIDENCE_THRESHOLDS:
                    prediction = routed_prediction(
                        experts[query_origins], probability, day, rule, confidence
                    )
                    score = mae(
                        prediction,
                        target[query_origins, part],
                        observed[query_origins, part],
                    )
                    rows.append({
                        "future_day": day + 1,
                        "max_leaf_nodes": leaves,
                        "min_samples_leaf": minimum,
                        "rule": rule,
                        "confidence": confidence,
                        "train_tail_mae": score,
                        "purge_steps": experts.shape[2],
                    })
                    if score < best_mae:
                        best_mae = score
                        best = {
                            "max_leaf_nodes": leaves,
                            "min_samples_leaf": minimum,
                            "rule": rule,
                            "confidence": confidence,
                            "train_tail_mae": score,
                            "purge_steps": experts.shape[2],
                        }
    assert best is not None
    center_mae = mae(
        experts[query_origins, 1, part],
        target[query_origins, part],
        observed[query_origins, part],
    )
    best["train_tail_center_mae"] = center_mae
    best["active"] = best_mae < center_mae
    return best, rows


def markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if column != "n":
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [
        "| " + " | ".join(display.columns) + " |",
        "|" + "|".join(["---"] * len(display.columns)) + "|",
    ]
    rows.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/benchmarks/knowair")
    parser.add_argument(
        "--input-cache",
        default="artifacts/knowair_quantile_router_eda/router_inputs.npz",
        help="Self-contained frozen expert/router tensors.",
    )
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Reproduce the 13.877 oracle ceiling without fitting the causal router.",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/knowair_quantile_router_eda"
    )
    args = parser.parse_args()

    data = KnowAirDataModule(args.data_root)
    inputs = np.load(args.input_cache)
    train_origins = inputs["train_forecast_start"].astype(np.int64)
    val_origins = inputs["val_forecast_start"].astype(np.int64)
    if val_origins.max() + 24 > data.boundaries["val"][1]:
        raise ValueError("validation origins cross into test")

    train_center = inputs["train_center"].astype(np.float32)
    val_center = inputs["val_center"].astype(np.float32)
    train_experts = expert_tensor(
        inputs["train_prediction_quantiles"], train_center
    )
    val_experts = expert_tensor(inputs["val_prediction_quantiles"], val_center)
    train_target = inputs["train_target"].astype(np.float32)
    val_target = inputs["val_target"].astype(np.float32)
    train_observed = inputs["train_observed"].astype(bool)
    val_observed = inputs["val_observed"].astype(bool)
    if args.oracle_only:
        rows, overall = oracle_selector_metrics(
            val_experts, val_target, val_observed
        )
        print(json.dumps({
            "oracle_station_day_mae": overall,
            "days": rows,
            "test_split_consulted": False,
        }, indent=2))
        return
    selected_rules, tuning_rows, metric_rows, val_predictions = [], [], [], []
    for day in range(3):
        print(f"Fitting quantile router day {day + 1}", flush=True)
        train_label, train_margin, train_valid = expert_labels(
            train_experts, train_target, train_observed, day
        )
        val_label, _, val_valid = expert_labels(
            val_experts, val_target, val_observed, day
        )
        train_features = router_features(
            data,
            train_origins,
            train_experts,
            inputs["train_prediction_std"].astype(np.float32),
            inputs["train_high_probability"].astype(np.float32),
            inputs["train_event_probability"].astype(np.float32),
            day,
        )
        val_features = router_features(
            data,
            val_origins,
            val_experts,
            inputs["val_prediction_std"].astype(np.float32),
            inputs["val_high_probability"].astype(np.float32),
            inputs["val_event_probability"].astype(np.float32),
            day,
        )
        selected, rows = fit_and_select(
            train_features,
            train_label,
            train_margin,
            train_valid,
            train_experts,
            train_target,
            train_observed,
            day,
            args.seed + day,
            args.origin_stride,
        )
        tuning_rows.extend(rows)
        selected_rules.append({"future_day": day + 1, **selected})
        full_origins = np.arange(0, len(train_features), args.origin_stride)
        x, y, chosen = flatten_selected(
            train_features, train_label, train_valid, full_origins
        )
        full_margin = train_margin[full_origins].reshape(-1)[chosen]
        classifier = build_classifier(
            int(selected["max_leaf_nodes"]),
            int(selected["min_samples_leaf"]),
            args.seed + day,
        )
        classifier.fit(
            x, y, sample_weight=1.0 + np.minimum(full_margin / 5.0, 4.0)
        )
        probability = expand_probability(classifier, val_features)
        part = slice(day * 8, (day + 1) * 8)
        center = val_experts[:, 1, part]
        oracle_choice = val_label
        oracle = np.take_along_axis(
            val_experts[:, :, part], oracle_choice[:, None, None, :], axis=1
        )[:, 0]
        if selected["active"]:
            prediction = routed_prediction(
                val_experts,
                probability,
                day,
                str(selected["rule"]),
                float(selected["confidence"]),
            )
        else:
            prediction = center.copy()
        val_predictions.append(prediction)
        mask = val_observed[:, part]
        predicted_label = np.argmax(probability, axis=2)
        valid_label = val_valid
        metric_rows.append({
            "future_day": day + 1,
            "center_mae": mae(center, val_target[:, part], mask),
            "routed_mae": mae(prediction, val_target[:, part], mask),
            "oracle_station_day_mae": mae(oracle, val_target[:, part], mask),
            "router_accuracy": float(
                (predicted_label[valid_label] == val_label[valid_label]).mean()
            ),
            "center_class_prevalence": float((val_label[valid_label] == 1).mean()),
            "low_class_prevalence": float((val_label[valid_label] == 0).mean()),
            "high_class_prevalence": float((val_label[valid_label] == 2).mean()),
            "active_from_train_gate": bool(selected["active"]),
            "n": int(mask.sum()),
        })

    prediction = np.concatenate(val_predictions, axis=1)
    metrics = pd.DataFrame(metric_rows)
    selected_frame = pd.DataFrame(selected_rules)
    tuning = pd.DataFrame(tuning_rows)
    overall_center = mae(val_center, val_target, val_observed)
    overall_router = mae(prediction, val_target, val_observed)
    promoted = bool(
        overall_router < overall_center
        and (metrics.loc[metrics.active_from_train_gate, "routed_mae"]
             <= metrics.loc[metrics.active_from_train_gate, "center_mae"]).all()
    )
    summary = {
        "boundary_csio_center_mae": overall_center,
        "quantile_router_mae": overall_router,
        "mae_gain": overall_center - overall_router,
        "oracle_station_day_mae": float(
            np.average(metrics.oracle_station_day_mae, weights=metrics.n)
        ),
        "promote_quantile_router": promoted,
        "information_setting": (
            "KnowAir past 72h + DART distribution + predicted event arrival; "
            "no realized future weather at validation"
        ),
        "test_split_consulted": False,
        "external_data_used": False,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "router_metrics.csv", index=False)
    selected_frame.to_csv(output / "train_selected_router.csv", index=False)
    tuning.to_csv(output / "train_tail_tuning.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "selected_val_predictions.npz",
        prediction=prediction.astype(np.float32),
        target=val_target,
        observed=val_observed,
        forecast_start=val_origins,
    )
    verdict = (
        "PASS — causal distribution router được promote."
        if promoted else
        "FAIL — oracle quantile ceiling không chuyển thành out-of-time gain."
    )
    report = f"""# KnowAir causal PM-distribution router

Router chỉ chọn giữa DART q10, Boundary-CSIO center và DART q90. Feature gồm
PM history, frozen predictive spread/regime probability và arrival probability
đã cross-fit từ weather history. Hyperparameter, hard/soft rule, confidence và
exact fallback đều chọn trên chronological train tail. Test không được đọc.

## Train-selected protection

{markdown(selected_frame)}

## Validation

{markdown(metrics)}

Overall center `{overall_center:.3f}`, routed `{overall_router:.3f}`, oracle
station-day `{summary['oracle_station_day_mae']:.3f}`. Khoảng cách lớn giữa
router và oracle đo trực tiếp phần branch identity còn không dự báo được.

**{verdict}**
"""
    (output / "REPORT_VI.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
