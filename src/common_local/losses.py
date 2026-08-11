"""Physical-mask-aligned compound L1 objective."""

from __future__ import annotations

import torch


def _masked_l1(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1)


def common_local_loss(output, target, pm_mean, pm_std):
    valid = target * pm_std + pm_mean >= 1e-4
    prediction = output["prediction"]
    central = _masked_l1(prediction, target, valid)
    common_target = target.mean(2)
    common = _masked_l1(output["common_prediction"], common_target, valid.any(2))
    residual_target = target - common_target[:, :, None]
    residual = _masked_l1(output["residual_prediction"], residual_target, valid)
    target_increment = torch.cat((
        target[:, :1] - output["persistence"][:, :1],
        target[:, 1:] - target[:, :-1],
    ), 1)
    predicted_increment = torch.cat((
        prediction[:, :1] - output["persistence"][:, :1],
        prediction[:, 1:] - prediction[:, :-1],
    ), 1)
    increment_valid = torch.cat((valid[:, :1], valid[:, 1:] & valid[:, :-1]), 1)
    increment = _masked_l1(predicted_increment, target_increment, increment_valid)
    total = central + .25 * common + .25 * residual + .10 * increment
    return total, {
        "central": float(central.detach()),
        "common": float(common.detach()),
        "residual": float(residual.detach()),
        "increment": float(increment.detach()),
    }

