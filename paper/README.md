# Paper package

This directory is the canonical paper-facing package after research freeze.

## Frozen scientific story

The retained method is the core-meteorology Transport--Source Recurrent Operator.
The paper's central evidence is:

1. recurrent state evolution improves validation MAE by 2.0990 over a
   parameter-matched direct model;
2. history-only forcing and memory studies plateau around 20.5--20.9 validation
   MAE, identifying future exogenous forcing uncertainty as the main limitation;
3. on the frozen UCI Beijing external test, the recurrent operator has lower MAE and
   sMAPE but higher RMSE than the matched direct model; temporal intervals still
   include zero, so this is suggestive rather than confirmatory evidence.

## Primary checkpoint provenance

The canonical checkpoints are in `checkpoints/tsr_primary/`. Their three-seed
validation MAE is `16.7630 +/- 0.0252`; their
KnowAir test MAE is `16.1266 +/- 0.0348`. The uniform ensemble reaches
`15.8205 / 24.3253 / 0.3721` for MAE/RMSE/sMAPE. The exact evaluation record is
`artifacts/tsr_primary_test.json`.

Both tracks consume realized target-period core meteorology. Comparisons with
AirDDE are comparisons of published point estimates under different information
availability, not identical-input or paired-significance claims.

## Contents

- `RESULTS.json`: canonical numbers and claim constraints.
- `CHECKLIST.md`: manuscript-writing and reporting checks.
- `artifacts/`: compact evidence supporting tables, ablations, temporal
  block-bootstrap analyses, the frozen Beijing replication, the audited GEFSv12
  operational protocol and the unopened China-AQI protocol.
- `checkpoints/`: canonical TSR checkpoints.

All superseded search artifacts, local AirDDE reproduction, broad EDA notebooks,
old China 24-to-6 results and rejected-branch checkpoints were removed during the
paper cleanup.
