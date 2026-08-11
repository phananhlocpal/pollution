"""Run the untouched AirDDE release with narrow modern-runtime compatibility shims."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
AIR_DDE = ROOT / "third_party/airdde"


def install_compatibility_shims():
    # PyTorch 2.8 removed the no-op `verbose` scheduler argument used by AirDDE.
    original = torch.optim.lr_scheduler.ReduceLROnPlateau

    def reduce_lr_on_plateau(*args, verbose=None, **kwargs):
        return original(*args, **kwargs)

    torch.optim.lr_scheduler.ReduceLROnPlateau = reduce_lr_on_plateau


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("mode", choices=("train", "eval"))
    known, passthrough = parser.parse_known_args()
    install_compatibility_shims()
    script = AIR_DDE / f"{known.mode}.py"
    os.chdir(AIR_DDE)
    sys.path.insert(0, str(AIR_DDE))
    sys.argv = [str(script), "--config_filename", "knowair_config.yaml", *passthrough]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

