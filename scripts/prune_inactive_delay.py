#!/usr/bin/env python3
"""Remove the inactive delay input columns from frozen no-delay TSR checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def prune_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("use_lagged_transport", True):
        raise ValueError(f"Checkpoint is not a no-delay model: {source}")
    state = checkpoint["model_state"]
    hidden_dim = int(config["hidden_dim"])

    recurrent = state["local_cell.weight_ih"]
    transport = state["transport_head.0.weight"]
    if recurrent.shape[1] < 2 or transport.shape[1] != hidden_dim + 3:
        raise ValueError(f"Unexpected legacy delay-slot shapes in {source}")

    # local input = [local features, current innovation, inactive delay]
    state["local_cell.weight_ih"] = recurrent[:, :-1].clone()
    # transport input = [hidden state, current innovation, inactive delay, wind]
    lag_column = hidden_dim + 1
    state["transport_head.0.weight"] = torch.cat(
        (transport[:, :lag_column], transport[:, lag_column + 1:]), dim=1
    ).clone()
    checkpoint["config"]["use_lagged_transport"] = False
    checkpoint["architecture"] = "transport_source_recurrent"
    checkpoint["pruned_inactive_delay_from"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prune_checkpoint(Path(args.source), Path(args.destination))


if __name__ == "__main__":
    main()
