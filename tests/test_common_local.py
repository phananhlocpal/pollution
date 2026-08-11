from pathlib import Path

import numpy as np
import pytest
import torch

from common_local.analog_memory import (
    global_keys, inverse_distance_weights, retrieve_neighbors, rolling_origin_folds,
)
from common_local.correction import FrozenResidualCorrection
from common_local.data import (
    CommonLocalOriginDataset, CommonLocalWindowDataset, FEATURES,
    GAGNNAirDDEWindowDataset, Panel,
    audit_gagnn_overlap, fit_seasonal_weather_weights, load_standard_panel,
)
from common_local.losses import common_local_loss
from common_local.metrics import validation_report
from common_local.model import CommonLocalForecaster
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.edge_time import corrected_mae, edge_time_features, fit_horizon_ridge


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


@pytest.mark.parametrize("mode", ["persistence", "learned", "latent", "factorized"])
def test_history_only_recurrent_does_not_read_future_weather(tmp_path, mode):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = TransportSourceRecurrentForecaster(
        city, stations=3, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, future_weather_mode=mode,
        weather_hidden_dim=5, use_auxiliary=False, use_month=False,
    ).eval()
    batch = {
        "x": torch.randn(2, 24, 3, 7),
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
    if mode == "latent":
        assert "weather_prediction" not in first
        assert model.use_lagged_transport is False
    if mode == "factorized":
        assert torch.allclose(first["weather_prediction"], second["weather_prediction"])
        assert model.use_lagged_transport is False
        changed_pm = {**batch, "x": batch["x"].clone()}
        changed_pm["x"][..., 0] += 1000
        with torch.no_grad():
            third = model(changed_pm)
        assert torch.allclose(first["weather_prediction"], third["weather_prediction"])
        assert model.transport_weather_head.out_features == 3
        assert model.source_forcing_cell.hidden_size > model.transport_forcing_cell.hidden_size


def test_gagnn_overlap_audit_and_split_local_96x24_reconstruction(tmp_path):
    samples, stations, features = 92, 209, 8
    timeline = np.arange(
        (samples + 23) * stations * features, dtype=np.float32
    ).reshape(samples + 23, stations, features)
    target_timeline = np.arange(
        (samples + 29) * stations, dtype=np.float32
    ).reshape(samples + 29, stations)
    timeline[..., 7] = target_timeline[:samples + 23]
    x = np.stack([timeline[i:i + 24] for i in range(samples)])
    y = np.stack([target_timeline[i + 24:i + 30] for i in range(samples)])
    for split in ("train", "val", "test"):
        np.save(tmp_path / f"{split}_x.npy", x)
        np.save(tmp_path / f"{split}_y.npy", y)
    report = audit_gagnn_overlap(tmp_path)
    assert report["reconstructable"] is True
    assert report["train"]["windows_96_to_24"] == 2

    metadata = type("Metadata", (), {
        "protocol": "96x24", "mean": np.zeros(8), "std": np.ones(8)
    })()
    dataset = GAGNNAirDDEWindowDataset(tmp_path, "train", metadata)
    first = dataset[0]
    assert len(dataset) == 2
    assert first["x"].shape == (96, 209, 8)
    assert first["y"].shape == (24, 209)
    np.testing.assert_array_equal(first["x"].numpy()[..., 0], target_timeline[:96])
    np.testing.assert_array_equal(first["y"].numpy(), target_timeline[96:120])


def test_seasonal_weather_modes_are_causal_and_preserve_daily_cycle(tmp_path):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    history = torch.arange(1 * 6 * 3 * 6, dtype=torch.float32).reshape(1, 6, 3, 6)
    common = dict(
        city_path=city, stations=3, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, seasonal_period=2,
        use_auxiliary=False, use_month=False,
    )
    seasonal = TransportSourceRecurrentForecaster(
        **common, future_weather_mode="seasonal"
    )
    expected = history[:, -2:].repeat(1, 2, 1, 1)
    assert torch.equal(seasonal._causal_future_weather(history), expected)

    weights = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (6, 1))
    weighted = TransportSourceRecurrentForecaster(
        **common, future_weather_mode="seasonal_weighted", seasonal_weights=weights
    )
    template = 0.25 * history[:, -2:] + 0.75 * history[:, -4:-2]
    assert torch.allclose(
        weighted._causal_future_weather(history), template.repeat(1, 2, 1, 1)
    )

    batch = {
        "x": torch.randn(2, 6, 3, 7),
        "future_weather": torch.randn(2, 4, 3, 6),
        "future_auxiliary": torch.randn(2, 4, 3, 5),
        "future_month": torch.randint(0, 12, (2, 4)),
    }
    changed = {**batch, "future_weather": batch["future_weather"] + 1000}
    for model in (seasonal.eval(), weighted.eval()):
        with torch.no_grad():
            first = model(batch)["prediction"]
            second = model(changed)["prediction"]
        assert torch.allclose(first, second)


def test_train_fitted_seasonal_weights_are_convex_and_share_wind_vector_weights():
    panel = _panel()
    weights = fit_seasonal_weather_weights(panel, period=8, cycles=3)
    assert weights.shape == (6, 3)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(weights[4], weights[5], atol=1e-7)


def test_rolling_analog_memory_never_reads_across_dev_boundary():
    folds = rolling_origin_folds(240, history=24, horizon=24, folds=3)
    assert len(folds) == 3
    for fold in folds:
        assert fold.candidate_origins.max() + 24 <= fold.query_origins.min()
        assert fold.query_origins.max() + 24 <= 240


def test_analog_retrieval_recovers_exact_regime_and_normalizes_weights():
    values = np.zeros((80, 2, 2), dtype=np.float32)
    values[:, :, 0] = np.arange(80)[:, None]
    values[:, :, 1] = (np.arange(80) % 7)[:, None]
    candidates = np.array((24, 32, 40), dtype=np.int64)
    keys = global_keys(values, candidates, 24, "multiscale")
    indices, distances = retrieve_neighbors(keys, keys[[1]], k=3)
    assert indices[0, 0] == 1
    weights = inverse_distance_weights(distances)
    np.testing.assert_allclose(weights.sum(1), 1.0)
    assert weights[0, 0] == pytest.approx(1.0)


def test_explicit_origin_dataset_and_adaptive_delay_remain_history_only(tmp_path):
    panel = _panel()
    dataset = CommonLocalOriginDataset(panel, np.array((48, 56)), horizon=4)
    assert int(dataset[0]["forecast_start"]) == 48
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = TransportSourceRecurrentForecaster(
        city, stations=3, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, future_weather_mode="factorized",
        use_adaptive_delay=True, delay_dim=5, use_auxiliary=False,
        use_month=False,
    ).eval()
    batch = {
        "x": torch.randn(2, 24, 3, 7),
        "future_weather": torch.randn(2, 4, 3, 6),
        "future_auxiliary": torch.randn(2, 4, 3, 5),
        "future_month": torch.randint(0, 12, (2, 4)),
    }
    with torch.no_grad():
        first = model(batch)
        second = model({**batch, "future_weather": batch["future_weather"] + 1000})
        diagnostic = model({**batch, "diagnostic_delay_attention": True})
    assert torch.allclose(first["prediction"], second["prediction"])
    assert model.use_lagged_transport is False
    assert diagnostic["delay_attention"].shape == (2, 4, 3, 24)
    assert torch.allclose(
        diagnostic["delay_attention"].sum(-1), torch.ones(2, 4, 3), atol=1e-6
    )


def test_edge_time_features_are_transport_only_and_ridge_recovers_signal():
    rng = np.random.default_rng(9)
    history = rng.normal(size=(5, 24, 4, 7)).astype(np.float32)
    prediction = rng.normal(size=(5, 3, 4)).astype(np.float32)
    coordinates = np.array(((100, 30), (101, 30), (100, 31), (101, 31)))
    features = edge_time_features(
        history, prediction, coordinates, np.zeros(7), np.ones(7),
        lags=(1, 2, 3, 4),
    )
    assert features.shape == (5, 3, 4, 4)
    np.testing.assert_allclose(features.mean(2), 0, atol=1e-6)
    coefficients = np.array((0.2, -0.1, 0.3, 0.05), dtype=np.float32)
    truth = prediction + np.einsum("bhnf,f->bhn", features, coefficients)
    valid = np.ones_like(truth, dtype=bool)
    fitted = fit_horizon_ridge(features, truth - prediction, valid, alpha=1e-8)
    baseline, corrected = corrected_mae(
        prediction, truth, features, fitted, target_mean=10, target_std=2
    )
    assert corrected < baseline * 1e-3


def test_global_source_memory_is_causal_and_attention_is_normalized(tmp_path):
    city = tmp_path / "city.txt"
    city.write_text("0 a 100 30\n1 b 101 31\n2 c 102 32\n")
    model = TransportSourceRecurrentForecaster(
        city, stations=3, horizon=4, hidden_dim=8, station_dim=3,
        month_dim=2, operator_dim=4, future_weather_mode="factorized",
        source_forcing_dim=8, global_source_memory_units=8,
        use_auxiliary=False, use_month=False,
    ).eval()
    batch = {
        "x": torch.randn(2, 24, 3, 7),
        "future_weather": torch.randn(2, 4, 3, 6),
        "future_auxiliary": torch.randn(2, 4, 3, 5),
        "future_month": torch.randint(0, 12, (2, 4)),
        "diagnostic_source_memory_attention": True,
    }
    with torch.no_grad():
        first = model(batch)
        second = model({**batch, "future_weather": batch["future_weather"] + 1000})
    assert torch.allclose(first["prediction"], second["prediction"])
    attention = first["source_memory_attention"]
    assert attention.shape == (2, 4, 8)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 4), atol=1e-6)


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
