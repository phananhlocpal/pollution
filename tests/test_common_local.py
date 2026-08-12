import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from common_local.correction import FrozenResidualCorrection
from common_local.data import (
    CommonLocalWindowDataset, FEATURES, GAGNNAirDDEWindowDataset, Panel,
    audit_gagnn_overlap, fit_seasonal_weather_weights, load_standard_panel,
)
from common_local.losses import common_local_loss
from common_local.metrics import validation_report
from common_local.model import CommonLocalForecaster
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.paired_statistics import (
    circular_block_bootstrap,
    origin_mae,
    summarize_paired_errors,
)


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


def test_origin_mae_preserves_forecast_origin_axis():
    truth = np.full((3, 2, 2), 10.0)
    prediction = truth + np.arange(3)[:, None, None]
    assert np.allclose(origin_mae(prediction, truth), [0.0, 1.0, 2.0])


def test_circular_block_bootstrap_is_reproducible_and_paired():
    difference = np.linspace(0.5, 1.5, 60)
    first = circular_block_bootstrap(difference, 7, replicates=100, seed=19)
    second = circular_block_bootstrap(difference, 7, replicates=100, seed=19)
    assert np.array_equal(first, second)
    summary = summarize_paired_errors(
        difference + 2.0, np.full(60, 2.0), block_lengths=(7,),
        replicates=100, seed=19,
    )
    assert summary["mean_mae_difference_reference_minus_proposed"] == pytest.approx(1.0)
    assert summary["block_sensitivity"]["7"]["ci95_percentile"][0] > 0.0


def test_all_retained_checkpoints_match_the_canonical_architecture():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "paper/CHECKPOINTS.json").read_text())
    for family in ("tsr_primary",):
        entry = manifest[family]
        assert sorted(entry["sha256"]) == [
            "seed_42.pt", "seed_43.pt", "seed_44.pt"
        ]
        for filename, expected_hash in entry["sha256"].items():
            checkpoint_path = root / entry.get(
                "path", f"paper/checkpoints/{family}"
            ) / filename
            assert checkpoint_path.exists(), f"missing retained checkpoint: {checkpoint_path}"
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            assert digest == expected_hash
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert checkpoint["architecture"] == "transport_source_recurrent"
            model = TransportSourceRecurrentForecaster(
                root / "data/benchmarks/knowair/city.txt",
                stations=184, **checkpoint["config"],
            )
            model.load_state_dict(checkpoint["model_state"])
            assert sum(parameter.numel() for parameter in model.parameters()) == 72_435


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
    if mode == "factorized":
        assert torch.allclose(first["weather_prediction"], second["weather_prediction"])
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
