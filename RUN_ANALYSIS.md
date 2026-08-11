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

## 5. Post-audit next-generation experiments

The KnowAir test has already been inspected. All next-generation model selection
therefore stays on train/validation; do not add `--split test` for these models.

Three-seed prediction ensembles (mean, median, and validation-fitted convex):

```powershell
.\.venv\Scripts\python.exe scripts\ensemble_predictions.py
```

Separated conservative transport and unconstrained source/sink correction:

```powershell
.\.venv\Scripts\python.exe -m common_local.ablate --variants transport_source --seeds 42 43 44 --epochs 20 --patience 4 --batch-size 256 --output-dir artifacts\transport_source --device cuda
```

Compact capacity/loss search with large-batch LR scheduling:

```powershell
.\.venv\Scripts\python.exe scripts\search_common_local.py --batch-size 256 --device cuda
```

Sequential Transport--Source Recurrent Operator (72,659 parameters):

```powershell
.\.venv\Scripts\python.exe -m common_local.train_dynamics --seeds 43 --epochs 30 --patience 6 --batch-size 256 --output-dir artifacts\transport_source_recurrent --device cuda
```

Core-meteorology future-forcing version (no PBL/ventilation/dewpoint-deficit/month):

```powershell
.\.venv\Scripts\python.exe -m common_local.train_dynamics --seeds 42 43 44 --epochs 30 --patience 6 --batch-size 256 --disable-auxiliary --disable-month --output-dir artifacts\transport_source_recurrent_strict --device cuda
```

Its validation-frozen convex ensemble reaches test MAE 15.8086 versus the AirDDE
paper reference 16.92. This still consumes realized future core meteorology and is
therefore not labelled exact input parity. Exact hashes, weights, and test results
are recorded under `frozen/transport_source_recurrent_strict`.

The optional regime/event expert is a warm-started ablation. It remains selected
by global validation MAE, not classification accuracy:

```powershell
.\.venv\Scripts\python.exe -m common_local.train_dynamics --seeds 43 --epochs 12 --patience 4 --batch-size 256 --lr 0.0003 --event-expert --initialize-from artifacts\transport_source_recurrent\seed_43\best_model.pt --output-dir artifacts\transport_source_recurrent_event --device cuda
```

Low-rank residual probe (training split only):

```powershell
.\.venv\Scripts\python.exe -m common_local.export_predictions --seed 43 --checkpoint artifacts\common_local\seed_43\best_model.pt --output artifacts\predictions\common_local_seed43_train --split train --batch-size 256
.\.venv\Scripts\python.exe scripts\probe_residual_modes.py artifacts\predictions\common_local_seed43_train
```

AirDDE multi-seed training is intentionally a separate long job. The wrapper now
retains each seed automatically instead of allowing the release's fixed checkpoint
path to overwrite the preceding seed:

```powershell
foreach ($seed in 42,43,44) {
  .\.venv\Scripts\python.exe scripts\run_airdde.py train --random_seed $seed
  .\.venv\Scripts\python.exe scripts\export_airdde_predictions.py --seed $seed --checkpoint "artifacts/airdde/seed_$seed/checkpoint.pth" --output "artifacts/predictions/airdde_seed${seed}_val"
}
```

An explicitly approximate paper-style run (Huber/SmoothL1 and patience 10) is
available with `--paper-style`; it is not labelled an exact paper reproduction.

## 6. Fairness and mechanism follow-up

History-only core-meteorology runs never read the realized future-weather tensor:

```powershell
.\.venv\Scripts\python.exe -m common_local.train_dynamics --seeds 43 --epochs 30 --patience 6 --batch-size 256 --disable-auxiliary --disable-month --future-weather-mode persistence --output-dir artifacts\transport_source_recurrent_history_persistence --device cuda
.\.venv\Scripts\python.exe -m common_local.train_dynamics --seeds 43 --epochs 30 --patience 6 --batch-size 256 --disable-auxiliary --disable-month --future-weather-mode learned --weather-hidden-dim 16 --weather-loss-weight 0.1 --output-dir artifacts\transport_source_recurrent_history_learned --device cuda
```

They obtained validation MAE 20.7945 and 20.7071, respectively, and were rejected
without opening their KnowAir test predictions.

The factorized history-only latent-forcing V2 was subsequently run for all three
seeds. It obtained `20.6533 / 20.8487 / 20.8613` validation MAE (mean
`20.7878 +/- 0.0952`) and failed the predeclared `<=18.0` gate. Its KnowAir test
and the corrected China-AQI 96-to-24 test both remain unopened.

The independently retrained validation ablation matrix (three seeds per variant)
is launched by:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrained_ablations.py
```

This covers no-transport, no-source/sink, no-lagged-transport, and a 72,630-parameter
direct counterpart against the 72,659-parameter recurrent model. Its summary is
separate from frozen-checkpoint knockout diagnostics.

The corrected external comparison is China-AQI only. Exact reconstruction,
history-only latent-forcing training, freeze, and evaluation commands are in
`EXTERNAL_REPLICATION.md`. The old 24h-to-6h GAGNN result is retained strictly as
an external generalization result and is not compared with AirDDE's 96h-to-24h
published metrics.
