# Paper package

This directory is the canonical paper-facing package after research freeze.

## Frozen scientific story

The retained method is the core-meteorology Transport--Source Recurrent Operator.
The paper's central evidence is:

1. recurrent state evolution improves validation MAE by 2.0952 over a
   parameter-matched direct model;
2. removing source/sink worsens MAE by 3.6224, while removing transport worsens it
   by 0.5132;
3. retraining without the explicit one-step lag changes MAE by -0.0038, so the
   final method specification omits that branch;
4. history-only forcing and memory studies plateau around 20.5--20.9 validation
   MAE, identifying future exogenous forcing uncertainty as the main limitation.

## Important result provenance

The reported KnowAir test headline `16.1285 +/- 0.0932` and uniform-ensemble result
`15.8130 / 24.3084 / 0.3718` come from the frozen core-meteorology checkpoints in
`checkpoints/core_meteo_lagged/`. Those checkpoints have
`use_lagged_transport=true`.

The no-lag architecture selected for the final method has three-seed validation
MAE `16.7630 +/- 0.0252`; its checkpoints are in
`checkpoints/core_meteo_no_lag/`. It has no frozen test result. Therefore the
existing headline test values must not be described as no-lag results unless a
separate, explicitly authorized evaluation resolves this mismatch.

Both tracks consume realized target-period core meteorology. Comparisons with
AirDDE are comparisons of published point estimates under different information
availability, not identical-input or paired-significance claims.

## Contents

- `RESULTS.json`: canonical numbers and claim constraints.
- `CHECKLIST.md`: manuscript-writing and reporting checks.
- `artifacts/`: compact evidence supporting tables, ablations, diagnostics and the
  unopened China-AQI protocol.
- `checkpoints/`: the six retained core-meteorology checkpoints.

All superseded search artifacts, local AirDDE reproduction, broad EDA notebooks,
old China 24-to-6 results and rejected-branch checkpoints were removed during the
paper cleanup.
