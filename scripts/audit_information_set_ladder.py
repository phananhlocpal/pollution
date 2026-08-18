"""E001 preflight: audit information sets before training a PM model.

This command intentionally reports no PM forecasting score.  It establishes
whether an archived meteorological forecast is a valid forecast-origin input and
quantifies its error against realized weather on the validation period.  A PM
model comparison is only valid after this preflight passes and the same frozen
architecture is run for every rung of the ladder.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from shift_pm import ForecastArchive, KnowAirDataModule


def validation_origins(
    data: KnowAirDataModule, history_steps: int, horizon: int, stride: int
) -> np.ndarray:
    if stride <= 0:
        raise ValueError("origin stride must be positive")
    left, right = data.boundaries["val"]
    origins = np.arange(max(left, history_steps), right - horizon + 1, stride, dtype=np.int64)
    if not len(origins):
        raise ValueError("validation split is too short for the requested horizon")
    if origins.max() + horizon > right:
        raise AssertionError("validation origins cross into test")
    return origins


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weather_error_summary(
    forecast: np.ndarray, realized: np.ndarray, feature_names: tuple[str, ...]
) -> dict[str, object]:
    absolute = np.abs(forecast.astype(np.float64) - realized.astype(np.float64))
    per_lead = absolute.mean(axis=(0, 2))
    result: dict[str, object] = {}
    for index, name in enumerate(feature_names):
        values = per_lead[:, index]
        result[name] = {
            "mae_by_lead": values.tolist(),
            "mae_day_1": float(values[:8].mean()),
            "mae_day_2": float(values[8:16].mean()) if len(values) >= 16 else None,
            "mae_day_3": float(values[16:24].mean()) if len(values) >= 24 else None,
        }
    return result


def markdown(report: dict[str, object]) -> str:
    forecast = report["forecast_weather"]
    lines = [
        "# E001 — Information-Set Ladder preflight",
        "",
        "This is a validation-only information audit, not a PM forecasting result.",
        "The test split was not indexed.",
        "",
        "## Fixed protocol",
        "",
        f"- KnowAir cadence: {report['cadence_hours']} h",
        f"- History / horizon: {report['history_steps']} / {report['horizon_steps']} steps",
        f"- Validation origins: {report['validation_origins']}",
        f"- Origin stride: {report['origin_stride']}",
        "- PM scaler and any eventual model selection must fit on training only.",
        "",
        "## Ladder status",
        "",
        "| rung | status | permitted information |",
        "|---|---|---|",
        "| PM history only | ready | PM at or before origin |",
        "| PM + past weather | ready | weather at or before origin |",
        "| weather persistence / climatology | ready | deterministic transform of past weather |",
        f"| archived forecast weather | {forecast['status']} | forecast published at or before origin |",
        "| realized future weather | diagnostic only | target-period observations; never operational |",
        "",
    ]
    if forecast["status"] == "ready":
        metadata = forecast["metadata"]
        lines.extend((
            "## Archived forecast audit",
            "",
            f"- Source / version: `{metadata['source']}` / `{metadata['model_version']}`",
            f"- Archive SHA-256: `{forecast['sha256']}`",
            f"- Features: {', '.join(forecast['weather_feature_names'])}",
            "- Assertions passed: coverage, unique origins, station count, finite values, "
            "issue time ≤ forecast origin, and exact valid-time/lead alignment.",
            "",
            "Per-feature forecast error is in `e001_preflight.json`; wind analysis in E002 must use "
            "the u/v vector jointly rather than treating direction as a scalar.",
        ))
    else:
        lines.extend((
            "## Blocker",
            "",
            "No archive was supplied. Obtain an archived NWP/AI forecast with issue time, valid time, "
            "source, and model version before making an operational claim. Synthetic persistence is a "
            "diagnostic rung only and must not be relabelled as an actual weather forecast.",
        ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/benchmarks/knowair")
    parser.add_argument("--forecast-weather", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/information_set_ladder"))
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--origin-stride", type=int, default=1)
    args = parser.parse_args()

    data = KnowAirDataModule(args.data_root)
    origins = validation_origins(
        data, args.history_steps, args.horizon, args.origin_stride
    )
    report: dict[str, object] = {
        "experiment": "E001_information_set_ladder_preflight",
        "split": "validation",
        "test_accessed": False,
        "cadence_hours": data.protocol.cadence_hours,
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon,
        "validation_origins": int(len(origins)),
        "origin_stride": args.origin_stride,
        "forecast_weather": {"status": "blocked_missing_archive"},
    }
    if args.forecast_weather:
        archive = ForecastArchive.load(args.forecast_weather)
        rows = archive.validate_for_validation(data, origins, args.horizon)
        forecast = archive.weather[rows]
        realized = data.observed_weather(
            origins, args.horizon, archive.weather_feature_names
        )
        report["forecast_weather"] = {
            "status": "ready",
            "path": str(args.forecast_weather),
            "sha256": file_sha256(args.forecast_weather),
            "metadata": archive.metadata,
            "weather_feature_names": list(archive.weather_feature_names),
            "forecast_error_mae": weather_error_summary(
                forecast, realized, archive.weather_feature_names
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e001_preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
