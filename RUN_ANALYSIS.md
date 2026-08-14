# Reproduction workflow

Research is frozen. The KnowAir ablation commands below are validation-only. The
Beijing test was opened once after its protocol and checkpoint hashes were frozen;
the China-AQI corrected test remains unopened.

## Environment check

```powershell
.\.venv\Scripts\python.exe scripts\verify_cuda.py
.\.venv\Scripts\python.exe -m pytest -q
```

## KnowAir validation ablations

### Full-history precursor diagnostic (H1--H4)

Repeat analogue matching while expanding history from PM + core meteorology to
all raw KnowAir meteorology and derived dynamical diagnostics. Future outcomes
are held fixed across levels and the test split is not accessed:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_full_history_precursors.py
```

This writes `artifacts/full_history_precursors/validation.json`.

The publication workflow retrains primary TSR, transport and source/sink
knockouts, and a parameter-matched direct baseline with seeds 42, 43 and 44.

```powershell
.\.venv\Scripts\python.exe scripts\run_retrained_ablations.py
```

Outputs are written under `artifacts/`; the canonical frozen numbers already used
by the manuscript are in `paper/RESULTS.json`.

## History-equivalent future-divergence diagnostic

Test whether validation origins with close train-history analogues still develop
different future meteorology and PM2.5 trajectories, without reading test:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_history_future_divergence.py
```

The primary result is written to
`artifacts/history_future_divergence/analysis.json`. Sensitivity runs can change
`--pca-components`; the checked analysis uses 64 components.

Evaluate K-nearest-history conditional medians, retrieved future-weather TSR
scenarios, best-of-K oracle bounds, and history-conditioned forecastability:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_analogue_scenarios.py --device cuda
```

This uses train futures as retrieval candidates, evaluates validation only, and
writes `artifacts/analogue_scenarios/validation.json`.

Evaluate the retained primary checkpoints on KnowAir with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tsr_primary.py --split test --allow-test --output paper/artifacts/tsr_primary_test.json --device cuda
```

## Forecasted-weather cascade

This validation-only experiment compares repeating the last observed weather
state with forecasting all 24 future weather steps using a causal GRU and feeding
those predictions into TSR. Neither branch reads observed target-period weather.

```powershell
.\.venv\Scripts\python.exe scripts\run_forecasted_weather_experiment.py --device cuda
```

For per-channel forecast skill and oracle diagnostics after the learned run:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_weather_forcing.py --checkpoint artifacts/forecasted_weather/learned/seed_43/best_model.pt --output artifacts/forecasted_weather/weather_diagnostics.json --device cuda
```

## UCI Beijing external replication

```powershell
.\.venv\Scripts\python.exe scripts\prepare_beijing_multisite.py
.\.venv\Scripts\python.exe -m common_local.train --panel-npz data/processed/beijing_multisite_3h.npz --expected-stations 12 --seeds 42 43 44 --epochs 30 --patience 6 --batch-size 256 --lr 0.003 --scheduler --hidden-dim 84 --horizon-dim 20 --station-dim 10 --dropout 0.1 --gru-layers 1 --loss l1 --output-dir artifacts/external_beijing/direct --device cuda
.\.venv\Scripts\python.exe -m common_local.train_dynamics --panel-npz data/processed/beijing_multisite_3h.npz --expected-stations 12 --seeds 42 43 44 --epochs 30 --patience 6 --batch-size 256 --disable-auxiliary --disable-month --output-dir artifacts/external_beijing/tsr_primary --device cuda
.\.venv\Scripts\python.exe scripts\evaluate_external_pair.py --split val --output paper/artifacts/beijing_external_validation_primary.json --device cuda
.\.venv\Scripts\python.exe scripts\evaluate_external_pair.py --split test --unlock-test --output paper/artifacts/beijing_external_test_primary.json --device cuda
```

The pre-test decision and checkpoint hashes are in
`paper/artifacts/beijing_external_freeze.json`. Re-running the last command is a
reproduction of an already-open test, not a new lockbox evaluation.

## Operational meteorology archive audit

```powershell
.\.venv\Scripts\python.exe scripts\audit_gefs_reforecast.py
```

The audit verifies the NOAA GEFSv12 3-hour leads and required 2-m temperature,
surface pressure, 2-m humidity and 100-m wind fields. It does not report an
operational forecasting result; ingestion and calibration remain incomplete.

## Residual diagnostics

```powershell
.\.venv\Scripts\python.exe scripts\summarize_residual_probes.py
.\.venv\Scripts\python.exe scripts\compare_residual_generations.py
.\.venv\Scripts\python.exe scripts\diagnose_weather_forcing.py
```

The compact retained diagnostic tables are in `paper/artifacts/`.

## China-AQI protocol

China-AQI is protocol-only. Prepare or audit the local GAGNN reconstruction, but
do not evaluate the corrected 96-to-24 test split.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_gagnn.py
.\.venv\Scripts\python.exe scripts\audit_gagnn_reconstruction.py
```

See `EXTERNAL_REPLICATION.md` for the reporting boundary.
