# Reproduction workflow

Research is frozen. These commands reproduce only the analyses retained for the
paper; they do not open either dataset's test split.

## Environment check

```powershell
.\.venv\Scripts\python.exe scripts\verify_cuda.py
.\.venv\Scripts\python.exe -m pytest -q
```

## KnowAir validation ablations

The publication workflow retrains the full recurrent operator, transport and
source/sink knockouts, the no-explicit-lag model, and a parameter-matched direct
baseline with seeds 42, 43 and 44.

```powershell
.\.venv\Scripts\python.exe scripts\run_retrained_ablations.py
```

Outputs are written under `artifacts/`; the canonical frozen numbers already used
by the manuscript are in `paper/RESULTS.json`.

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
