"""Train-only rolling residual probe for physics-conditioned edge×time delays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common_local.analog_memory import rolling_origin_folds
from common_local.data import CommonLocalOriginDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.edge_time import corrected_mae, edge_time_features, fit_horizon_ridge
from common_local.train import _loader, choose_device, move_batch


ALPHAS = (0.01, 0.1, 1.0)
ADVANCEMENT_GAIN = 0.3


def _subsample(origins: np.ndarray, maximum: int) -> np.ndarray:
    if len(origins) <= maximum:
        return origins
    return origins[np.linspace(0, len(origins) - 1, maximum, dtype=int)]


def predict(model, dataset, panel, device, batch_size):
    histories, predictions, truths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, 43, False):
            histories.append(batch["x"].numpy())
            batch = move_batch(batch, device)
            predictions.append(model(batch)["prediction"].cpu().numpy())
            truths.append(batch["y"].cpu().numpy())
    history = np.concatenate(histories)
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    features = edge_time_features(
        history, prediction, panel.coordinates, panel.mean, panel.std
    )
    valid = truth * float(panel.std[0]) + float(panel.mean[0]) >= 1e-4
    return prediction, truth, features, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/edge_time_probe/decision.json")
    parser.add_argument("--max-fit-origins", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.root)
    panel = load_panel(root)
    device = choose_device(args.device)
    rows = []
    for fold_index, fold in enumerate(rolling_origin_folds(panel.split_points[0]), 1):
        checkpoint_path = root / (
            f"artifacts/rolling_delay/factorized_v3/fold_{fold_index}/"
            "seed_43/best_model.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = TransportSourceRecurrentForecaster(
            root / "data/benchmarks/knowair/city.txt", stations=len(panel.stations),
            coordinates=panel.coordinates, **checkpoint["config"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        fit_set = CommonLocalOriginDataset(
            panel, _subsample(fold.candidate_origins, args.max_fit_origins)
        )
        dev_set = CommonLocalOriginDataset(panel, fold.query_origins)
        train_prediction, train_truth, train_features, train_valid = predict(
            model, fit_set, panel, device, args.batch_size
        )
        dev_prediction, dev_truth, dev_features, _ = predict(
            model, dev_set, panel, device, args.batch_size
        )
        residual = train_truth - train_prediction
        for alpha in ALPHAS:
            coefficients = fit_horizon_ridge(
                train_features, residual, train_valid, alpha
            )
            baseline, corrected = corrected_mae(
                dev_prediction, dev_truth, dev_features, coefficients,
                float(panel.mean[0]), float(panel.std[0]),
            )
            rows.append({
                "fold": fold_index, "alpha": alpha,
                "fit_origins": len(fit_set), "dev_origins": len(dev_set),
                "baseline_mae": baseline, "corrected_mae": corrected,
                "gain": baseline - corrected,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = []
    for alpha in ALPHAS:
        selected = [row for row in rows if row["alpha"] == alpha]
        gains = [row["gain"] for row in selected]
        aggregate.append({
            "alpha": alpha,
            "mean_baseline_mae": float(np.mean([row["baseline_mae"] for row in selected])),
            "mean_corrected_mae": float(np.mean([row["corrected_mae"] for row in selected])),
            "mean_gain": float(np.mean(gains)),
            "minimum_fold_gain": float(np.min(gains)),
            "all_folds_improve": bool(np.all(np.asarray(gains) > 0)),
        })
    selected = max(aggregate, key=lambda row: row["mean_gain"])
    advance = bool(
        selected["mean_gain"] >= ADVANCEMENT_GAIN and selected["all_folds_improve"]
    )
    payload = {
        "protocol": "V3 residual ridge fitted/evaluated on train-internal rolling folds",
        "candidate_space": "8 geographic neighbors x lags 1/2/3/4",
        "physics_prior": "distance + source-wind alignment + travel-time mismatch",
        "correction_constraint": "zero mean across stations (transport only)",
        "fold_results": rows,
        "aggregate": aggregate,
        "selected": selected,
        "advancement_gate": "mean gain >=0.3 MAE and positive gain in every fold",
        "advance_edge_time_v4": advance,
        "original_validation_accessed": False,
        "test_accessed": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
