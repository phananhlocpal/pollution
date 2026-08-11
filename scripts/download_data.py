#!/usr/bin/env python3
"""Download and extract the official UCI Beijing Multi-Site dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)
EXPECTED_SITES = 12


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract(outer_zip: Path, raw_dir: Path) -> Path:
    target = raw_dir / "PRSA_Data_20130301-20170228"
    if len(list(target.glob("*.csv"))) == EXPECTED_SITES:
        return target

    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(outer_zip) as archive:
            nested_name = next(
                name for name in archive.namelist() if name.endswith("PRSA2017_Data_20130301-20170228.zip")
            )
            archive.extract(nested_name, temp)
        with zipfile.ZipFile(temp / nested_name) as nested:
            nested.extractall(raw_dir)

    files = list(target.glob("*.csv"))
    if len(files) != EXPECTED_SITES:
        raise RuntimeError(f"Expected {EXPECTED_SITES} site CSVs, found {len(files)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    archive = args.raw_dir / "beijing_multisite.zip"
    if args.force or not archive.exists():
        print(f"Downloading {URL}")
        download(URL, archive)
    target = extract(archive, args.raw_dir)
    print(f"Ready: {target} ({len(list(target.glob('*.csv')))} stations)")


if __name__ == "__main__":
    main()

