"""Convert official AirDDE eval arrays into the unified bundle format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--prediction", default="third_party/airdde/logs/prediction.npy")
    parser.add_argument("--truth", default="third_party/airdde/logs/truths.npy")
    parser.add_argument("--output", default="artifacts/predictions/airdde_repro_test")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    root = Path(args.root); output = root / args.output; output.mkdir(parents=True, exist_ok=True)
    prediction = np.load(root / args.prediction, mmap_mode="r")
    truth = np.load(root / args.truth, mmap_mode="r")
    if prediction.shape != truth.shape or prediction.shape[1:] != (24, 184):
        raise ValueError(f"Unexpected AirDDE arrays: {prediction.shape}, {truth.shape}")
    shutil.copy2(root / args.prediction, output / "prediction.npy")
    shutil.copy2(root / args.truth, output / "truth.npy")
    raw_path = root / "data/benchmarks/knowair/KnowAir.npy"
    raw = np.load(raw_path, mmap_mode="r")
    test_left = int(len(raw) * .75)
    starts = test_left + 24 + np.arange(len(prediction), dtype=np.int64)
    expected = len(raw) - test_left - 48 + 1
    if len(starts) != expected:
        raise ValueError(f"AirDDE produced {len(starts)} origins; canonical test has {expected}")
    np.save(output / "forecast_start.npy", starts)
    persistence = np.lib.format.open_memmap(
        output / "persistence.npy", mode="w+", dtype="float32", shape=prediction.shape
    )
    persistence[:] = np.asarray(raw[starts - 1, :, -1], dtype=np.float32)[:, None, :]
    persistence.flush()
    commit = (root / "artifacts/airdde/reproduction_manifest.json")
    airdde = json.loads(commit.read_text()) if commit.exists() else {}
    manifest = {
        "model": "AirDDE-repro", "seed": args.seed, "split": "test",
        "shape": list(prediction.shape), "cadence_hours": 3,
        "dataset_sha256": sha256(raw_path),
        "station_order_sha256": sha256(root / "data/benchmarks/knowair/city.txt"),
        "airdde_commit": airdde.get("airdde_commit"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

