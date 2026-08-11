"""Download the four additional air-quality benchmarks used by the EDA.

The downloader is idempotent, resumes ordinary HTTP downloads, verifies known
file sizes, and extracts the two archives.  AirFormer's public archive is the
official tiny reproducibility sample; the paper's >500 GB dataset is not public.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import gdown
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmarks"

HTTP_FILES = {
    "beijing_kdd": [
        (
            "kdd_cup_2018_dataset_with_missing_values.zip",
            "https://zenodo.org/api/records/4656719/files/"
            "kdd_cup_2018_dataset_with_missing_values.zip/content",
            2_456_948,
        ),
        (
            "beijing_station_coords.csv",
            "https://raw.githubusercontent.com/decisionintelligence/"
            "Air-DualODE/main/dataset/Beijing1718/station.csv",
            1_010,
        ),
    ],
    "airformer": [
        (
            "data.zip",
            "https://raw.githubusercontent.com/yoshall/AirFormer/main/data/data.zip",
            84_374_630,
        )
    ],
    "airqualitybench": [
        (
            "aq_compact_2021.h5",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "aq_compact_2021.h5?download=true",
            186_061_095,
        ),
        (
            "aq_compact_2022.h5",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "aq_compact_2022.h5?download=true",
            179_894_888,
        ),
        (
            "aq_compact_2023.h5",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "aq_compact_2023.h5?download=true",
            277_680_451,
        ),
        (
            "aq_compact_2024.h5",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "aq_compact_2024.h5?download=true",
            317_830_115,
        ),
        (
            "aq_compact_2025.h5",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "aq_compact_2025.h5?download=true",
            324_894_267,
        ),
        (
            "adj_mx_10.pkl",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "adj_mx_10.pkl?download=true",
            55_409_047,
        ),
        (
            "scaler.csv",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "scaler.csv?download=true",
            161,
        ),
        (
            "selected_nodes_metadata.csv",
            "https://huggingface.co/datasets/xuxing123/aq_dataset/resolve/main/"
            "selected_nodes_metadata.csv?download=true",
            101_712,
        ),
    ],
}


def download_http(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = destination.stat().st_size if destination.exists() else 0
    if current == expected_size:
        print(f"verified {destination.relative_to(ROOT)}")
        return
    if current > expected_size:
        raise RuntimeError(f"{destination} is larger than the official file")

    headers = {"Range": f"bytes={current}-"} if current else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        supports_resume = response.status_code == 206
        mode = "ab" if current and supports_resume else "wb"
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=2**20):
                if chunk:
                    handle.write(chunk)
    actual = destination.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"Incomplete {destination}: expected {expected_size}, received {actual}"
        )


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)


def download_dataset(name: str) -> None:
    directory = BASE / name
    directory.mkdir(parents=True, exist_ok=True)
    if name == "knowair":
        target = directory / "KnowAir.npy"
        expected_size = 309_685_376
        if target.exists() and target.stat().st_size != expected_size:
            raise RuntimeError(
                f"{target} has {target.stat().st_size} bytes; expected {expected_size}"
            )
        if not target.exists():
            gdown.download(
                id="1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L",
                output=str(target),
                quiet=False,
                resume=True,
            )
        if target.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete KnowAir download: {target}")
        print(f"verified {target.relative_to(ROOT)}")
        download_http(
            "https://raw.githubusercontent.com/shuowang-ai/PM2.5-GNN/main/data/city.txt",
            directory / "city.txt",
            7_216,
        )
        return

    for filename, url, expected_size in HTTP_FILES[name]:
        download_http(url, directory / filename, expected_size)
    if name == "beijing_kdd":
        extract_zip(
            directory / "kdd_cup_2018_dataset_with_missing_values.zip", directory
        )
    elif name == "airformer":
        extract_zip(directory / "data.zip", directory / "extracted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["beijing_kdd", "knowair", "airformer", "airqualitybench"],
        default=["beijing_kdd", "knowair", "airformer", "airqualitybench"],
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        print(f"\n[{dataset}]")
        download_dataset(dataset)


if __name__ == "__main__":
    main()
