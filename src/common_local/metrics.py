"""Physical-unit validation metrics."""

from __future__ import annotations

import numpy as np


PERIODS = {
    "day1_1_24h": slice(0, 8),
    "day2_25_48h": slice(8, 16),
    "day3_49_72h": slice(16, 24),
    "overall_1_72h": slice(0, 24),
}


def physical_metrics(prediction, truth):
    truth = truth.copy(); truth[truth < 1e-4] = 0
    valid = truth != 0; difference = prediction - truth
    denominator = (np.abs(truth) + np.abs(prediction)) / 2 + 1e-8
    return {
        "mae": float(np.abs(difference)[valid].mean()),
        "rmse": float(np.sqrt(np.square(difference)[valid].mean())),
        "smape": float(np.mean(np.abs(difference) / denominator)),
        "smape_masked": float((np.abs(difference) / denominator)[valid].mean()),
        "points": int(valid.sum()),
    }


def validation_report(prediction, truth):
    report = {
        name: physical_metrics(prediction[:, selection], truth[:, selection])
        for name, selection in PERIODS.items()
    }
    report["horizon_hours"] = list(range(3, 73, 3))
    report["mae_by_horizon"] = [
        physical_metrics(prediction[:, h:h + 1], truth[:, h:h + 1])["mae"]
        for h in range(24)
    ]
    return report

