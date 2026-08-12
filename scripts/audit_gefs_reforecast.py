#!/usr/bin/env python3
"""Audit the NOAA GEFSv12 fields needed for an operational TSR experiment."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


BUCKET = "https://noaa-gefs-retrospective.s3.amazonaws.com"
VARIABLES = {
    "temperature_2m": ("tmp_2m", "TMP:2 m above ground"),
    "surface_pressure": ("pres_sfc", "PRES:surface"),
    "specific_humidity_2m": ("spfh_2m", "SPFH:2 m above ground"),
    "u_wind_100m": ("ugrd_hgt", "UGRD:100 m above ground"),
    "v_wind_100m": ("vgrd_hgt", "VGRD:100 m above ground"),
}


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TSR-reproducibility-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def content_length(url: str) -> int:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "TSR-reproducibility-audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", default="2015010100")
    parser.add_argument("--maximum-lead", type=int, default=93)
    parser.add_argument(
        "--output", default="paper/artifacts/gefs_v12_operational_protocol.json"
    )
    args = parser.parse_args()
    year = args.initialization[:4]
    prefix = (
        f"GEFSv12/reforecast/{year}/{args.initialization}/c00/Days:1-10"
    )
    expected_leads = list(range(3, args.maximum_lead + 1, 3))
    fields, total_bytes = {}, 0
    for name, (stem, selector) in VARIABLES.items():
        object_url = f"{BUCKET}/{prefix}/{stem}_{args.initialization}_c00.grib2"
        index_url = object_url + ".idx"
        object_bytes = content_length(object_url)
        lines = read_url(index_url).decode("utf-8").strip().splitlines()
        records = []
        for index, line in enumerate(lines):
            parts = line.split(":", 2)
            offset = int(parts[1])
            next_offset = int(lines[index + 1].split(":", 2)[1]) if index + 1 < len(lines) else object_bytes
            match = re.search(r":(\d+) hour fcst:", line)
            if selector in line and match:
                lead = int(match.group(1))
                if lead <= args.maximum_lead:
                    records.append({
                        "lead_hour": lead,
                        "byte_start": offset,
                        "byte_end_inclusive": next_offset - 1,
                        "bytes": next_offset - offset,
                    })
        leads = [record["lead_hour"] for record in records]
        if leads != expected_leads:
            raise ValueError(f"{name}: expected leads {expected_leads}, got {leads}")
        selected_bytes = sum(record["bytes"] for record in records)
        total_bytes += selected_bytes
        fields[name] = {
            "selector": selector,
            "grib_url": object_url,
            "index_url": index_url,
            "messages": len(records),
            "selected_message_bytes": selected_bytes,
            "full_object_bytes": object_bytes,
            "first_record": records[0],
            "last_record": records[-1],
        }

    payload = {
        "status": "archive_audited_not_ingested",
        "source": "NOAA GEFSv12 reforecast on the NOAA Open Data Dissemination S3 bucket",
        "source_documentation": "https://registry.opendata.aws/noaa-gefs-reforecast/",
        "initialization_audited": args.initialization,
        "member": "c00 control",
        "availability": "one initialization per day for 2000--2019; five members",
        "required_leads_hours": expected_leads,
        "lead_alignment": (
            "Use the latest daily 00 UTC initialization. Origins from 00 through 21 UTC "
            "require leads through 72--93 hours for a 72-hour target."
        ),
        "field_mapping": {
            "temperature": "2-m temperature",
            "pressure": "surface pressure",
            "relative_humidity": "derive from 2-m specific humidity, pressure, and temperature",
            "wind": "100-m u/v components; convert to speed and circular direction",
        },
        "fields": fields,
        "selected_message_bytes_per_initialization": total_bytes,
        "estimated_selected_message_gib_per_365_day_year": total_bytes * 365 / 2**30,
        "estimated_selected_message_gib_for_four_years": total_bytes * 365 * 4 / 2**30,
        "experiment_completed": False,
        "reason_not_reported_as_result": (
            "The forecast archive has been verified, but the four-year regional extraction, "
            "meteorological calibration, and model rerun have not been completed."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
