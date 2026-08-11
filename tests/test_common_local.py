from pathlib import Path

import numpy as np
import torch

from common_local.data import CommonLocalWindowDataset, FEATURES, Panel
from common_local.losses import common_local_loss
from common_local.metrics import validation_report
from common_local.model import CommonLocalForecaster


def _panel():
    physical = np.arange(240 * 3 * 7, dtype=float).reshape(240, 3, 7) + 1
    train = physical[:120].reshape(-1, 7)
    mean, std = train.mean(0), train.std(0)
    return Panel(((physical - mean) / std).astype(np.float32), physical,
                 (120, 180), mean, std, ["a", "b", "c"], FEATURES)


def test_windows_stay_inside_chronological_split():
    panel = _panel(); dataset = CommonLocalWindowDataset(panel, "val")
    sample = dataset[0]
    assert int(sample["forecast_start"]) == 144
    assert int(dataset.starts[-1]) + 48 == panel.split_points[1]
    assert sample["x"].shape == (24, 3, 7)
    assert sample["future_weather"].shape == (24, 3, 6)


def test_model_preserves_zero_mean_local_prediction():
    model = CommonLocalForecaster(
        stations=3, horizon=4, hidden_dim=8, horizon_dim=4,
        station_dim=3, dropout=0,
    )
    output = model({"x": torch.randn(2, 6, 3, 7),
                    "future_weather": torch.randn(2, 4, 3, 6)})
    assert output["prediction"].shape == (2, 4, 3)
    assert torch.allclose(output["residual_prediction"].mean(-1),
                          torch.zeros(2, 4), atol=1e-6)


def test_compound_l1_objective_retains_component_weights():
    target = torch.tensor([[[0., 2.], [3., 4.]]])
    common = target.mean(2); prediction = target + 1
    output = {
        "prediction": prediction, "common_prediction": common + 1,
        "residual_prediction": target - common[:, :, None] + 1,
        "persistence": torch.zeros_like(target),
    }
    loss, parts = common_local_loss(output, target, 0, 1)
    expected = (parts["central"] + .25 * parts["common"]
                + .25 * parts["residual"] + .10 * parts["increment"])
    assert torch.isclose(loss, torch.tensor(expected))


def test_validation_report_has_three_days_and_24_horizons():
    truth = np.ones((2, 24, 3)) * 10; prediction = truth + 2
    report = validation_report(prediction, truth)
    assert report["overall_1_72h"]["mae"] == 2
    assert len(report["mae_by_horizon"]) == 24


def test_all_retained_checkpoints_match_the_canonical_architecture():
    root = Path(__file__).resolve().parents[1]
    for seed in (42, 43, 44):
        checkpoint = torch.load(
            root / "artifacts/common_local" / f"seed_{seed}" / "best_model.pt",
            map_location="cpu", weights_only=False,
        )
        CommonLocalForecaster().load_state_dict(checkpoint["model_state"], strict=True)
