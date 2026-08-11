# Execution plan from the residual-driven audit

All commands below are run from the repository root in PowerShell. The test split
remains guarded until the architecture and residual corrections are frozen.

## 1. Recreate the missing retained checkpoints (long)

The manifest records hashes for three old checkpoints, but those files are not in
Git or the current workspace. Recreate them with:

```powershell
.\.venv\Scripts\python.exe -m common_local.train --epochs 20 --patience 5 --batch-size 8 --device cuda --seeds 42 43 44
```

Do not compare the resulting validation number with AirDDE's paper test number.

## 2. Frozen-model validation predictions and residual screening

Run this once per seed (seed 42 shown):

```powershell
.\.venv\Scripts\python.exe -m common_local.export_predictions --seed 42 --checkpoint artifacts\common_local\seed_42\best_model.pt --output artifacts\predictions\common_local_seed42_val
.\.venv\Scripts\python.exe -m benchmarking.evaluator artifacts\predictions\common_local_seed42_val --output artifacts\predictions\common_local_seed42_val\evaluation.json
.\.venv\Scripts\python.exe -m benchmarking.residual_probe artifacts\predictions\common_local_seed42_val --output artifacts\residual_probe\seed_42
```

The probe covers horizons 3/6/12/24/36/48/72 hours and writes:

- residual correlation for neighbour innovation/trend, regional factors,
  wind-aligned current and 3h/6h lagged innovation;
- dewpoint, precipitation, PBL height, ventilation and dewpoint deficit;
- purged chronological ridge corrections and incremental MAE versus a
  current-PM + frozen-prediction correction;
- station/region, season, PM-bin and extreme-event phase summaries.

Only signals with stable incremental validation gain across seeds should proceed to
neural ablations.

## 3. AirDDE released-code reproduction (long)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_airdde.py
.\.venv\Scripts\python.exe scripts\run_airdde.py train --random_seed 2024
```

This is `AirDDE-repro`, not the paper's 16.92 number. The pinned release uses MAE
and patience 3, while the paper describes Huber and tolerance 10. Keep a separate
checkpoint/output directory for every seed before starting the next seed because
the official code uses the fixed `checkpoints/airdde` path.

## 4. Final test only after architecture freeze

Export common_local with the explicit test guard:

```powershell
.\.venv\Scripts\python.exe -m common_local.export_predictions --seed 42 --checkpoint artifacts\common_local\seed_42\best_model.pt --output artifacts\predictions\common_local_seed42_test --split test --allow-test
```

Evaluate the matching AirDDE checkpoint and import its official arrays:

```powershell
.\.venv\Scripts\python.exe scripts\run_airdde.py eval --random_seed 2024
.\.venv\Scripts\python.exe scripts\import_airdde_predictions.py --seed 2024
.\.venv\Scripts\python.exe -m benchmarking.evaluator artifacts\predictions\common_local_seed42_test --compare artifacts\predictions\airdde_repro_test --block-length 24 --bootstrap-repetitions 2000 --output artifacts\predictions\paired_test.json
```

The evaluator rejects mismatched shapes, truth arrays or forecast origins and
reports Day 1/2/3/overall MAE, RMSE, sMAPE, persistence-relative MAE skill, and a
paired moving-block bootstrap interval for the MAE difference.

