import numpy as np

from scripts.eda_knowair_quantile_router import (
    expert_labels,
    oracle_selector_metrics,
    purged_train_tail_indices,
    routed_prediction,
)


def test_expert_labels_choose_station_day_mae_winner() -> None:
    experts = np.zeros((2, 3, 8, 2), dtype=np.float32)
    experts[:, 0] = 1.0
    experts[:, 1] = 3.0
    experts[:, 2] = 5.0
    target = np.empty((2, 8, 2), dtype=np.float32)
    target[..., 0] = 1.2
    target[..., 1] = 4.8
    observed = np.ones_like(target, dtype=bool)
    label, margin, valid = expert_labels(experts, target, observed, day=0)
    np.testing.assert_array_equal(label, [[0, 2], [0, 2]])
    assert (margin > 0).all()
    assert valid.all()


def test_hard_router_uses_requested_expert() -> None:
    experts = np.stack([
        np.full((1, 8, 2), 1.0),
        np.full((1, 8, 2), 2.0),
        np.full((1, 8, 2), 3.0),
    ], axis=1)
    probability = np.array([[[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]]])
    prediction = routed_prediction(
        experts, probability, day=0, rule="hard", confidence=0.5
    )
    np.testing.assert_allclose(prediction[..., 0], 1.0)
    np.testing.assert_allclose(prediction[..., 1], 3.0)


def test_low_confidence_has_exact_center_fallback() -> None:
    experts = np.stack([
        np.full((1, 8, 1), 1.0),
        np.full((1, 8, 1), 2.0),
        np.full((1, 8, 1), 3.0),
    ], axis=1)
    probability = np.full((1, 1, 3), 1 / 3)
    prediction = routed_prediction(
        experts, probability, day=0, rule="soft", confidence=0.6
    )
    np.testing.assert_allclose(prediction, 2.0)


def test_oracle_selector_aggregates_day_winners() -> None:
    experts = np.zeros((1, 3, 24, 1), dtype=np.float32)
    experts[:, 0] = 1.0
    experts[:, 1] = 2.0
    experts[:, 2] = 3.0
    target = np.concatenate((
        np.full((1, 8, 1), 1.1),
        np.full((1, 8, 1), 2.1),
        np.full((1, 8, 1), 2.9),
    ), axis=1)
    rows, overall = oracle_selector_metrics(
        experts, target, np.ones_like(target, dtype=bool)
    )
    assert len(rows) == 3
    np.testing.assert_allclose(overall, 0.1, atol=1e-6)


def test_train_tail_purge_prevents_target_overlap() -> None:
    fitted, tuned = purged_train_tail_indices(
        length=400, horizon=24, origin_stride=4
    )
    assert fitted.max() + 24 <= tuned.min()
