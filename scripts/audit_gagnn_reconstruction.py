"""Verify whether official GAGNN windows exactly reconstruct China-AQI 96->24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common_local.data import audit_gagnn_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--gagnn-dir", default="data/benchmarks/china_aqi_gagnn")
    parser.add_argument(
        "--output", default="artifacts/external_replication/china_aqi_reconstruction.json"
    )
    args = parser.parse_args()
    root = Path(args.root)
    report = audit_gagnn_overlap(root / args.gagnn_dir)
    report.update({
        "dataset": "official GAGNN China-AQI release",
        "reconstructed_protocol": {"history": 96, "horizon": 24, "cadence_hours": 1},
        "split_policy": "each released split reconstructed independently; boundaries never joined",
        "future_covariates": "historical factors only; no realized target-period meteorology",
    })
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["reconstructable"]:
        raise SystemExit("Exact reconstruction failed; do not create a 96x24 benchmark")


if __name__ == "__main__":
    main()
