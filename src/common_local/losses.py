"""Physical-mask-aligned compound L1 objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_l1(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1)


def _masked_error(prediction, target, mask, kind="l1"):
    if kind == "l1":
        error = torch.abs(prediction - target)
    elif kind == "huber":
        error = F.smooth_l1_loss(prediction, target, reduction="none", beta=.5)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    mask = mask.to(prediction.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def common_local_loss(output, target, pm_mean, pm_std, kind="l1"):
    valid = target * pm_std + pm_mean >= 1e-4
    prediction = output["prediction"]
    central = _masked_error(prediction, target, valid, kind)
    common_target = target.mean(2)
    common = _masked_error(output["common_prediction"], common_target, valid.any(2), kind)
    residual_target = target - common_target[:, :, None]
    residual = _masked_error(output["residual_prediction"], residual_target, valid, kind)
    target_increment = torch.cat((
        target[:, :1] - output["persistence"][:, :1],
        target[:, 1:] - target[:, :-1],
    ), 1)
    predicted_increment = torch.cat((
        prediction[:, :1] - output["persistence"][:, :1],
        prediction[:, 1:] - prediction[:, :-1],
    ), 1)
    increment_valid = torch.cat((valid[:, :1], valid[:, 1:] & valid[:, :-1]), 1)
    increment = _masked_error(predicted_increment, target_increment, increment_valid, kind)
    total = central + .25 * common + .25 * residual + .10 * increment
    return total, {
        "central": float(central.detach()),
        "common": float(common.detach()),
        "residual": float(residual.detach()),
        "increment": float(increment.detach()),
    }

