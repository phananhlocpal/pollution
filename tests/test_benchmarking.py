import numpy as np

from benchmarking.evaluator import origin_mae, paired_block_interval


def test_origin_mae_uses_airdde_truth_mask():
    truth = np.array([[[0.0, 10.0]], [[20.0, 30.0]]])
    prediction = truth + 2
    assert np.allclose(origin_mae(prediction, truth), [2.0, 2.0])


def test_paired_block_interval_reports_signed_delta():
    result = paired_block_interval(np.ones(48) * 2, np.ones(48),
                                   block_length=24, repetitions=100, seed=1)
    assert result["mean_delta_mae_a_minus_b"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
