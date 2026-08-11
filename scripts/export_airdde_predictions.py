"""Export a pinned AirDDE checkpoint into the unified benchmark bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from run_airdde import AIR_DDE, ROOT, install_compatibility_shims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", default="third_party/airdde/checkpoints/airdde/checkpoint.pth")
    parser.add_argument("--output", default="artifacts/predictions/airdde_seed2024_val")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test access without --allow-test.")

    os.chdir(AIR_DDE)
    sys.path.insert(0, str(AIR_DDE))
    install_compatibility_shims()
    from eval import Evaluation_Air_Pollution
    from utils import ConfigDict, fix_seed, load_config

    config = ConfigDict(load_config("knowair_config.yaml"))
    config.data.batch_size = args.batch_size
    namespace = argparse.Namespace(
        random_seed=args.seed, num_nodes=184, input_dim=6, output_dim=1,
        horizon=24, rnn_units=64, num_rnn_layers=1, model_name="AirDDE",
        report_filepath=str(ROOT / "artifacts/airdde"), exp_idx=0,
    )
    for name, value in config.items():
        setattr(namespace, name, value)
    namespace.to_log_file = False
    namespace.to_stdout = False
    namespace.GPU.use_gpu = torch.cuda.is_available() and namespace.GPU.use_gpu
    namespace.GPU.gpu = 0
    fix_seed(args.seed)
    experiment = Evaluation_Air_Pollution(namespace)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location=experiment.device, weights_only=True)
    experiment.model.load_state_dict(checkpoint, strict=True)
    dataset, loader = experiment._get_data(args.split)
    experiment.model.eval()

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(dataset), 24, 184)
    prediction = np.lib.format.open_memmap(output_dir / "prediction.npy", "w+", dtype="float32", shape=shape)
    truth = np.lib.format.open_memmap(output_dir / "truth.npy", "w+", dtype="float32", shape=shape)
    cursor = 0
    with torch.inference_mode():
        for x, target in loader:
            x, target, future = experiment._prepare_data(x, target)
            predicted = experiment.model(x, future, target, batches_seen=0)
            batch_prediction = dataset.inverse_transform(predicted.cpu().permute(1, 0, 2)).numpy()
            batch_truth = dataset.inverse_transform(target.cpu().permute(1, 0, 2)).numpy()
            size = len(batch_prediction)
            prediction[cursor:cursor + size] = batch_prediction
            truth[cursor:cursor + size] = batch_truth
            cursor += size
    prediction.flush(); truth.flush()

    raw = np.load(ROOT / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
    left = int(len(raw) * (.5 if args.split == "val" else .75))
    starts = left + 24 + np.arange(len(dataset), dtype=np.int64)
    np.save(output_dir / "forecast_start.npy", starts)
    persistence = np.lib.format.open_memmap(
        output_dir / "persistence.npy", "w+", dtype="float32", shape=shape
    )
    persistence[:] = np.asarray(raw[starts - 1, :, -1], dtype=np.float32)[:, None, :]
    persistence.flush()
    reproduction = json.loads((ROOT / "artifacts/airdde/reproduction_manifest.json").read_text())
    manifest = {
        "model": "AirDDE-repro", "seed": args.seed, "split": args.split,
        "shape": list(shape), "cadence_hours": 3, "history_steps": 24,
        "horizon_steps": 24, "dataset_sha256": reproduction["knowair_sha256"],
        "station_order_sha256": reproduction["city_order_sha256"],
        "graph_data_sha256": reproduction["graph_data_sha256"],
        "airdde_commit": reproduction["airdde_commit"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

