"""Construct AirDDE and run one small CUDA forward/backward batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from run_airdde import AIR_DDE, install_compatibility_shims


def main():
    os.chdir(AIR_DDE)
    sys.path.insert(0, str(AIR_DDE))
    install_compatibility_shims()
    from trainer import Exp_Air
    from utils import ConfigDict, fix_seed, load_config

    config = ConfigDict(load_config("knowair_config.yaml"))
    config.data.batch_size = 2
    args = argparse.Namespace(
        random_seed=2024, num_nodes=184, input_dim=6, output_dim=1,
        horizon=24, rnn_units=64, num_rnn_layers=1, model_name="AirDDE",
        to_log_file=False, to_stdout=False, exp_idx=0,
    )
    for name, value in config.items():
        setattr(args, name, value)
    args.GPU.use_gpu = torch.cuda.is_available() and args.GPU.use_gpu
    args.GPU.gpu = 0
    fix_seed(args.random_seed)
    experiment = Exp_Air(args)
    _, loader = experiment._get_data("train")
    x, target = next(iter(loader))
    x, target, future = experiment._prepare_data(x, target)
    output = experiment.model(x, future, target, batches_seen=0)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    result = {
        "input_shape": list(x.shape), "output_shape": list(output.shape),
        "loss": float(loss.detach()), "finite": bool(torch.isfinite(output).all()),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

