"""Export aligned prediction/truth tensors from a retained common_local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CommonLocalWindowDataset, load_panel
from .correction import FrozenResidualCorrection, FrozenTransportSourceCorrection
from .dynamics import TransportSourceRecurrentForecaster
from .model import CommonLocalForecaster


RECURRENT_ARCHITECTURES = {
    "transport_source_recurrent",
    "latent_forcing_transport_source_recurrent_v2",
    "factorized_exogenous_transport_source_v3",
    "latent_impact_distillation_tsr",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args):
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test access without --allow-test (freeze the architecture first).")
    root = Path(args.root)
    panel = load_panel(root)
    dataset = CommonLocalWindowDataset(panel, args.split, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint_path = root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    components = tuple(checkpoint.get("components", ()))
    architecture = checkpoint.get("architecture", "static_correction" if components else "common_local")
    if architecture in RECURRENT_ARCHITECTURES:
        model = TransportSourceRecurrentForecaster(
            root / "data/benchmarks/knowair/city.txt",
            stations=len(panel.stations), **checkpoint.get("config", {}),
        ).to(device)
    elif architecture == "transport_source":
        model = FrozenTransportSourceCorrection(
            root / "data/benchmarks/knowair/city.txt", panel.mean, panel.std
        ).to(device)
    elif components:
        model = FrozenResidualCorrection(
            components, root / "data/benchmarks/knowair/city.txt", panel.mean, panel.std
        ).to(device)
    else:
        model = CommonLocalForecaster(
            stations=len(panel.stations), **checkpoint.get("config", {})
        ).to(device)
    incompatible = model.load_state_dict(
        checkpoint["model_state"], strict=architecture not in RECURRENT_ARCHITECTURES
    )
    if architecture in RECURRENT_ARCHITECTURES:
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if key != "station_threshold"]
        if missing or unexpected:
            raise ValueError(f"Incompatible recurrent checkpoint: missing={missing}, unexpected={unexpected}")
    model.eval()

    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(dataset), 24, len(panel.stations))
    prediction = np.lib.format.open_memmap(output / "prediction.npy", mode="w+", dtype="float32", shape=shape)
    truth = np.lib.format.open_memmap(output / "truth.npy", mode="w+", dtype="float32", shape=shape)
    persistence = np.lib.format.open_memmap(output / "persistence.npy", mode="w+", dtype="float32", shape=shape)
    starts = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            size = len(batch["y"])
            device_batch = {key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in batch.items()}
            pred = model(device_batch)["prediction"].cpu().numpy()
            pred = pred * panel.std[0] + panel.mean[0]
            target = batch["y"].numpy() * panel.std[0] + panel.mean[0]
            last = batch["x"][:, -1, :, 0].numpy() * panel.std[0] + panel.mean[0]
            prediction[cursor:cursor + size] = pred
            truth[cursor:cursor + size] = target
            persistence[cursor:cursor + size] = last[:, None, :]
            starts[cursor:cursor + size] = batch["forecast_start"].numpy()
            cursor += size
    prediction.flush(); truth.flush(); persistence.flush()
    np.save(output / "forecast_start.npy", starts)
    manifest = {
        "model": architecture if architecture in RECURRENT_ARCHITECTURES else
                 "common_local+" + architecture if components else "common_local",
        "architecture": architecture,
        "components": list(components), "seed": args.seed, "split": args.split,
        "shape": list(shape), "cadence_hours": panel.cadence_hours,
        "history_steps": 24, "horizon_steps": 24,
        "dataset_sha256": sha256(root / "data/benchmarks/knowair/KnowAir.npy"),
        "station_order_sha256": sha256(root / "data/benchmarks/knowair/city.txt"),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="artifacts/common_local/seed_42/best_model.pt")
    parser.add_argument("--output", default="artifacts/predictions/common_local_seed42_val")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
