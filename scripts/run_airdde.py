"""Run the untouched AirDDE release with narrow modern-runtime compatibility shims."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import shutil
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
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument("--paper-style", action="store_true",
                        help="Approximate the paper loss/patience (Huber, patience 10).")
    known, passthrough = parser.parse_known_args()
    install_compatibility_shims()
    script = AIR_DDE / f"{known.mode}.py"
    config_path = AIR_DDE / "knowair_config.yaml"
    artifact_family = "airdde"
    original_l1 = torch.nn.L1Loss
    if known.paper_style:
        artifact_family = "airdde_paper_style"
        generated = ROOT / "artifacts/airdde_paper_style/knowair_paper_style.yaml"
        generated.parent.mkdir(parents=True, exist_ok=True)
        text = config_path.read_text().replace("criterion: mae", "criterion: huber")
        text = text.replace("patience: 3", "patience: 10")
        generated.write_text(text)
        config_path = generated
        # The release trainer falls back to L1 for every criterion except mse/mae.
        # Redirect that fallback to SmoothL1 without changing the pinned checkout.
        torch.nn.L1Loss = torch.nn.SmoothL1Loss
    official_checkpoint = AIR_DDE / "checkpoints/airdde/checkpoint.pth"
    retained_checkpoint = ROOT / f"artifacts/{artifact_family}/seed_{known.random_seed}/checkpoint.pth"
    if known.mode == "eval":
        if not retained_checkpoint.exists():
            raise SystemExit(f"Missing retained checkpoint: {retained_checkpoint}")
        official_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(retained_checkpoint, official_checkpoint)
    os.chdir(AIR_DDE)
    sys.path.insert(0, str(AIR_DDE))
    sys.argv = [str(script), "--config_filename", str(config_path),
                "--random_seed", str(known.random_seed), *passthrough]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        torch.nn.L1Loss = original_l1
    if known.mode == "train":
        retained_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(official_checkpoint, retained_checkpoint)
        print(f"Retained {artifact_family} seed {known.random_seed}: {retained_checkpoint}")


if __name__ == "__main__":
    main()
