"""Benchmark frozen model inference on the same CUDA device and batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np

from run_airdde import AIR_DDE, ROOT, install_compatibility_shims
from common_local.correction import FrozenResidualCorrection, FrozenTransportSourceCorrection
from common_local.data import CommonLocalWindowDataset, load_panel
from common_local.dynamics import TransportSourceRecurrentForecaster
from common_local.model import CommonLocalForecaster


def timed(name, model, call, batch_size, runs):
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        events = []
        for _ in range(runs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record(); events.append((start, end))
        torch.cuda.synchronize()
    samples = np.asarray([start.elapsed_time(end) for start, end in events])
    return {
        "model": name, "batch_size": batch_size, "runs": runs,
        "milliseconds_per_batch_median": float(np.median(samples)),
        "milliseconds_per_batch_p95": float(np.quantile(samples, .95)),
        "milliseconds_per_origin_median": float(np.median(samples) / batch_size),
        "milliseconds_per_origin_p95": float(np.quantile(samples, .95) / batch_size),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args(); device = torch.device("cuda")
    panel = load_panel(ROOT); dataset = CommonLocalWindowDataset(panel, "val", args.batch_size)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(args.batch_size)])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    baseline = CommonLocalForecaster().to(device)
    baseline.load_state_dict(torch.load(ROOT / "artifacts/common_local/seed_43/best_model.pt", map_location=device, weights_only=False)["model_state"])
    rows = [timed("common_local", baseline, lambda: baseline(batch), args.batch_size, args.runs)]
    del baseline; torch.cuda.empty_cache()
    selected = FrozenResidualCorrection(
        ("wind", "meteo"), ROOT / "data/benchmarks/knowair/city.txt", panel.mean, panel.std
    ).to(device)
    selected.load_state_dict(torch.load(
        ROOT / "artifacts/correction_ablation/wind_meteo/seed_43/best_model.pt",
        map_location=device, weights_only=False,
    )["model_state"])
    rows.append(timed("common_local+wind+meteo", selected, lambda: selected(batch), args.batch_size, args.runs))
    del selected; torch.cuda.empty_cache()
    split = FrozenTransportSourceCorrection(
        ROOT / "data/benchmarks/knowair/city.txt", panel.mean, panel.std
    ).to(device)
    split.load_state_dict(torch.load(
        ROOT / "artifacts/transport_source/transport_source/seed_43/best_model.pt",
        map_location=device, weights_only=False,
    )["model_state"])
    rows.append(timed("transport_source_correction", split, lambda: split(batch), args.batch_size, args.runs))
    del split; torch.cuda.empty_cache()
    dynamics_checkpoint = torch.load(
        ROOT / "artifacts/transport_source_recurrent/seed_43/best_model.pt",
        map_location=device, weights_only=False,
    )
    dynamics = TransportSourceRecurrentForecaster(
        ROOT / "data/benchmarks/knowair/city.txt", **dynamics_checkpoint["config"]
    ).to(device)
    incompatible = dynamics.load_state_dict(dynamics_checkpoint["model_state"], strict=False)
    if [key for key in incompatible.missing_keys if key != "station_threshold"] or incompatible.unexpected_keys:
        raise ValueError(f"Incompatible recurrent checkpoint: {incompatible}")
    rows.append(timed("transport_source_recurrent", dynamics, lambda: dynamics(batch), args.batch_size, args.runs))
    del dynamics, batch; torch.cuda.empty_cache()

    os.chdir(AIR_DDE); sys.path.insert(0, str(AIR_DDE)); install_compatibility_shims()
    from eval import Evaluation_Air_Pollution
    from utils import ConfigDict, fix_seed, load_config
    config = ConfigDict(load_config("knowair_config.yaml")); config.data.batch_size = args.batch_size
    namespace = argparse.Namespace(random_seed=2024, num_nodes=184, input_dim=6, output_dim=1,
        horizon=24, rnn_units=64, num_rnn_layers=1, model_name="AirDDE",
        report_filepath=str(ROOT / "artifacts/airdde"), exp_idx=0)
    for name, value in config.items(): setattr(namespace, name, value)
    namespace.to_log_file = False; namespace.to_stdout = False
    namespace.GPU.use_gpu = True; namespace.GPU.gpu = 0; fix_seed(2024)
    experiment = Evaluation_Air_Pollution(namespace)
    experiment.model.load_state_dict(torch.load(
        ROOT / "artifacts/airdde/seed_2024/checkpoint.pth",
        map_location=device, weights_only=True,
    ))
    _, loader = experiment._get_data("val")
    x, target = next(iter(loader)); x, target, future = experiment._prepare_data(x, target)
    rows.append(timed("AirDDE-repro", experiment.model,
                      lambda: experiment.model(x, future, target, batches_seen=0),
                      args.batch_size, args.runs))
    output = ROOT / "artifacts/final_analysis/efficiency.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "results": rows}, indent=2))
    print(json.dumps({"device": torch.cuda.get_device_name(0), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
