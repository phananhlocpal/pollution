from pathlib import Path

import numpy as np
import pytest
import torch

from common_local.correction import FrozenResidualCorrection
from common_local.data import CommonLocalWindowDataset, FEATURES, Panel, load_standard_panel
from common_local.losses import common_local_loss
from common_local.metrics import validation_report
from common_local.model import CommonLocalForecaster
from common_local.dynamics import TransportSourceRecurrentForecaster


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
    assert report["overall_1_72h"]["smape_masked"] == report["overall_1_72h"]["smape"]


def test_validation_report_supports_external_six_hour_horizon():
    truth = np.ones((2, 6, 3)) * 10
    report = validation_report(truth + 2, truth, cadence_hours=1)
    assert report["overall_1_6h"]["mae"] == 2
    assert report["horizon_hours"] == [1, 2, 3, 4, 5, 6]


def test_all_retained_checkpoints_match_the_canonical_architecture():
    root = Path(__file__).resolve().parents[1]
    checkpoints = [
        root / "artifacts/common_local" / f"seed_{seed}" / "best_model.pt"
        for seed in (42, 43, 44)
    ]
    if not any(path.exists() for path in checkpoints):
        pytest.skip("checkpoints are generated artifacts and are not present")
    assert all(path.exists() for path in checkpoints), "retained checkpoint set is incomplete"
    for seed, checkpoint_path in zip((42, 43, 44), checkpoints):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu", weights_only=False,
        )
        CommonLocalForecaster().load_state_dict(checkpoint["model_state"], strict=True)


def test_frozen_correction_starts_as_exact_baseline(tmp_path):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = FrozenResidualCorrection(("spatial",), city, np.zeros(7), np.ones(7), hidden_dim=4)
    batch = {"x": torch.randn(2, 6, 3, 7), "future_weather": torch.randn(2, 24, 3, 6)}
    model.base = CommonLocalForecaster(stations=3, hidden_dim=48)
    model.eval()
    with torch.no_grad():
        baseline = model.base(batch)["prediction"]
        corrected = model(batch)["prediction"]
    assert torch.allclose(corrected, baseline)


def test_recurrent_operator_is_sequential_and_conservative_at_initialization(tmp_path):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = TransportSourceRecurrentForecaster(
        city, stations=3, horizon=4, hidden_dim=8,
        station_dim=3, month_dim=2, operator_dim=4,
    )
    batch = {
        "x": torch.randn(2, 6, 3, 7),
        "future_weather": torch.randn(2, 4, 3, 6),
        "future_auxiliary": torch.randn(2, 4, 3, 5),
        "future_month": torch.randint(0, 12, (2, 4)),
    }
    output = model(batch)
    persistence = batch["x"][:, -1, :, 0][:, None].expand(-1, 4, -1)
    assert output["prediction"].shape == (2, 4, 3)
    assert torch.allclose(output["prediction"], persistence, atol=1e-6)
    assert torch.allclose(
        output["transport_operator"].mean(-1), torch.zeros(2, 4), atol=1e-6
    )


@pytest.mark.parametrize("mode", ["persistence", "learned"])
def test_history_only_recurrent_does_not_read_future_weather(tmp_path, mode):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = TransportSourceRecurrentForecaster(
        city, stations=3, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, future_weather_mode=mode,
        weather_hidden_dim=5, use_auxiliary=False, use_month=False,
    ).eval()
    batch = {
        "x": torch.randn(2, 6, 3, 7),
        "future_weather": torch.randn(2, 4, 3, 6),
        "future_auxiliary": torch.randn(2, 4, 3, 5),
        "future_month": torch.randint(0, 12, (2, 4)),
    }
    changed = {**batch, "future_weather": batch["future_weather"] + 1000}
    with torch.no_grad():
        first = model(batch)
        second = model(changed)
    assert torch.allclose(first["prediction"], second["prediction"])
    if mode == "learned":
        assert torch.allclose(first["weather_prediction"], second["weather_prediction"])


def test_external_panel_contract_and_operator_port(tmp_path):
    path = tmp_path / "external.npz"
    rng = np.random.default_rng(4)
    np.savez(
        path, target=rng.normal(size=(240, 3)),
        weather=rng.normal(size=(240, 3, 7)),
        coordinates=np.array([[100., 30.], [101., 31.], [102., 32.]]),
        station_ids=np.array(["a", "b", "c"]),
    )
    panel = load_standard_panel(path)
    assert panel.split_points == (168, 192)
    assert panel.values.shape == (240, 3, 8)
    model = TransportSourceRecurrentForecaster(
        stations=3, weather_dim=7, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, coordinates=panel.coordinates,
        use_auxiliary=False, use_month=False,
    )
    sample = CommonLocalWindowDataset(panel, "train", max_samples=2)[0]
    batch = {key: value[None] for key, value in sample.items()}
    output = model(batch)
    assert output["prediction"].shape == (1, 4, 3)
    with pytest.raises(ValueError, match="Expected 209 stations"):
        load_standard_panel(path, expected_stations=209)
