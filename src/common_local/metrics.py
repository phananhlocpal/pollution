"""Physical-unit validation metrics."""

from __future__ import annotations

import numpy as np


PERIODS = {
    "day1_1_24h": slice(0, 8),
    "day2_25_48h": slice(8, 16),
    "day3_49_72h": slice(16, 24),
    "overall_1_72h": slice(0, 24),
}


def physical_metrics(prediction, truth, valid_mask=None):
    truth = truth.copy(); truth[truth < 1e-4] = 0
    valid = truth != 0; difference = prediction - truth
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)
    denominator = (np.abs(truth) + np.abs(prediction)) / 2 + 1e-8
    symmetric_error = np.abs(difference) / denominator
    return {
        "mae": float(np.abs(difference)[valid].mean()),
        "rmse": float(np.sqrt(np.square(difference)[valid].mean())),
        "mape": float((np.abs(difference)[valid] / np.abs(truth)[valid]).mean() * 100),
        "smape": float(symmetric_error[valid].mean() if valid_mask is not None else symmetric_error.mean()),
        "smape_masked": float(symmetric_error[valid].mean()),
        "points": int(valid.sum()),
    }


def validation_report(prediction, truth, cadence_hours=3, valid_mask=None):
    horizon = prediction.shape[1]
    if horizon == 24 and cadence_hours == 3:
        selections = PERIODS
    else:
        selections = {f"overall_1_{horizon * cadence_hours}h": slice(0, horizon)}
    report = {
        name: physical_metrics(
            prediction[:, selection], truth[:, selection],
            None if valid_mask is None else valid_mask[:, selection],
        )
        for name, selection in selections.items()
    }
    report["horizon_hours"] = list(range(cadence_hours, horizon * cadence_hours + 1, cadence_hours))
    report["mae_by_horizon"] = [
        physical_metrics(
            prediction[:, h:h + 1], truth[:, h:h + 1],
            None if valid_mask is None else valid_mask[:, h:h + 1],
        )["mae"]
        for h in range(horizon)
    ]
    return report
