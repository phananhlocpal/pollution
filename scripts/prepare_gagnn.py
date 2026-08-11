"""Fetch and verify the official GAGNN China-AQI release used by AirDDE."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

import gdown
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Friger/GAGNN.git"
COMMIT = "509ac7d6eb55914979fc45f6d23e967021cfd270"
DRIVE_ID = "1I_vpbLJhOJpNh-TpLdSWsaG3xCpzMVSQ"
ARCHIVE_SHA256 = "4decea9aaafcd60e08e495083d708da238ab84ba0b507d42fe692c7405b52fd8"
REQUIRED = {
    "train_x.npy": (14259, 24, 209, 8),
    "val_x.npy": (2037, 24, 209, 8),
    "test_x.npy": (4074, 24, 209, 8),
    "train_y.npy": (14259, 6, 209),
    "val_y.npy": (2037, 6, 209),
    "test_y.npy": (4074, 6, 209),
    "loc_filled.npy": (209, 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    checkout = ROOT / "third_party/gagnn"
    if not checkout.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(checkout)], check=True)
    current = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if current != COMMIT:
        raise SystemExit(f"GAGNN checkout is {current}; expected pinned {COMMIT}")
    archive = ROOT / "data/downloads/gagnn_dataset"
    destination = ROOT / "data/benchmarks/china_aqi_gagnn"
    if args.force_download or not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=DRIVE_ID, output=str(archive), quiet=False)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("Official GAGNN archive hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    if any(not (destination / name).exists() for name in REQUIRED):
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    for name, expected in REQUIRED.items():
        array = np.load(destination / name, mmap_mode="r", allow_pickle=True)
        if array.shape != expected:
            raise SystemExit(f"{name}: got {array.shape}, expected {expected}")
    print(f"Verified official GAGNN China-AQI at {destination}")
    print("Protocol: 209 cities, 24 hourly history -> 6 hourly forecasts, 70/10/20 split")


if __name__ == "__main__":
    main()
