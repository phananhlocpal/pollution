"""Probe low-rank residual structure and causally available mode forecastability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--output", default="artifacts/residual_modes/probe.json")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--horizon-index", type=int, default=23)
    args = parser.parse_args()
    bundle = Path(args.bundle)
    prediction = np.load(bundle / "prediction.npy", mmap_mode="r")
    truth = np.load(bundle / "truth.npy", mmap_mode="r")
    split = int(len(prediction) * 2 / 3)
    fit_residual = np.asarray(
        truth[:split] - prediction[:split], dtype=np.float32
    ).reshape(-1, prediction.shape[-1])
    fit_residual[~np.isfinite(fit_residual)] = 0
    station_mean = fit_residual.mean(0, keepdims=True)
    centered = fit_residual - station_mean
    svd = TruncatedSVD(n_components=args.components, n_iter=7, random_state=2026)
    svd.fit(centered)
    cumulative = np.cumsum(svd.explained_variance_ratio_)

    horizon_residual = np.asarray(
        truth[:, args.horizon_index] - prediction[:, args.horizon_index], dtype=np.float32
    )
    coefficients = (horizon_residual - station_mean) @ svd.components_.T
    # At an origin, the 72h residual from the origin 24 steps ago has just become
    # observable. Lags 24/48 therefore avoid using future truth.
    lag1, lag2 = 24, 48
    indices = np.arange(lag2, len(coefficients))
    x = np.concatenate((coefficients[indices - lag1], coefficients[indices - lag2]), 1)
    y = coefficients[indices]
    train = indices < split
    test = ~train
    ridge = Ridge(alpha=10.0).fit(x[train], y[train])
    predicted_coefficients = ridge.predict(x[test])
    residual_prediction = predicted_coefficients @ svd.components_ + station_mean
    selected = indices[test]
    base_prediction = np.asarray(prediction[selected, args.horizon_index])
    selected_truth = np.asarray(truth[selected, args.horizon_index])
    valid = selected_truth >= 1e-4
    base_mae = float(np.abs(base_prediction - selected_truth)[valid].mean())
    corrected_mae = float(np.abs(base_prediction + residual_prediction - selected_truth)[valid].mean())
    result = {
        "bundle": str(bundle), "fit_origins": split,
        "holdout_origins": int(test.sum()), "horizon_hours": (args.horizon_index + 1) * 3,
        "causal_origin_lags": [lag1, lag2],
        "cumulative_explained_variance": {
            str(k): float(cumulative[k - 1]) for k in (1, 4, 8, 16, 32)
            if k <= len(cumulative)
        },
        "holdout_base_mae": base_mae,
        "holdout_mode_corrected_mae": corrected_mae,
        "holdout_delta_mae": corrected_mae - base_mae,
        "decision": "continue" if corrected_mae < base_mae else "reject",
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
