"""Prepare the pinned AirDDE checkout without duplicating the 310 MB KnowAir tensor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AIR_DDE = ROOT / "third_party/airdde"
SOURCE = ROOT / "data/benchmarks/knowair/KnowAir.npy"
TARGET = AIR_DDE / "datasets/KnowAir/KnowAir.npy"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not (AIR_DDE / ".git").exists():
        raise SystemExit("Initialize AirDDE first: git submodule update --init --recursive")
    if not SOURCE.exists():
        raise SystemExit("KnowAir.npy is missing; run scripts/download_benchmarks.py --datasets knowair")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    method = "existing"
    if not TARGET.exists():
        try:
            os.link(SOURCE, TARGET)
            method = "hardlink"
        except OSError:
            shutil.copy2(SOURCE, TARGET)
            method = "copy"
    source_hash = sha256(SOURCE)
    if sha256(TARGET) != source_hash:
        raise SystemExit("AirDDE KnowAir.npy does not match the canonical data snapshot")
    commit = subprocess.check_output(
        ["git", "-C", str(AIR_DDE), "rev-parse", "HEAD"], text=True
    ).strip()
    raw = np.load(SOURCE, mmap_mode="r")
    train_end, validation_end = int(len(raw) * .5), int(len(raw) * .75)
    graph_path = AIR_DDE / "datasets/KnowAir/graph_data.npz"
    graph = np.load(graph_path)
    if graph["adj_mx"].shape != (raw.shape[1], raw.shape[1]):
        raise SystemExit("AirDDE graph node count does not match KnowAir.npy")
    payload = {
        "airdde_commit": commit,
        "airdde_repository": "https://github.com/w2obin/airdde-aaai.git",
        "knowair_sha256": source_hash,
        "knowair_shape": list(raw.shape),
        "city_order_sha256": sha256(ROOT / "data/benchmarks/knowair/city.txt"),
        "graph_data_sha256": sha256(graph_path),
        "split_indices": {"train_end": train_end, "validation_end": validation_end},
        "forecast_origin_counts": {
            "train": train_end - 48 + 1,
            "validation": validation_end - train_end - 48 + 1,
            "test": len(raw) - validation_end - 48 + 1,
        },
        "data_materialization": method,
        "official_config": "third_party/airdde/knowair_config.yaml",
        "known_release_difference": "Released config uses MAE and patience=3; paper describes Huber and tolerance=10.",
        "station_csv_note": "The 1498-row station.csv is loaded but not used by the released model; graph_data.npz has 184 aligned nodes.",
    }
    output = ROOT / "artifacts/airdde"
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction_manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
