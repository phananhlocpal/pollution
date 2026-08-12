# Paper package

This directory is the canonical paper-facing package after research freeze.

## Frozen scientific story

The retained method is the core-meteorology Transport--Source Recurrent Operator.
The paper's central evidence is:

1. recurrent state evolution improves validation MAE by 2.0990 over a
   parameter-matched direct model;
2. in the delayed candidate matrix, removing source/sink worsens MAE by 3.6224,
   while removing transport worsens it by 0.5132;
3. adding an explicit one-step lag to primary TSR worsens MAE by 0.0038; paired
   block-bootstrap intervals show no benefit, so the final model omits it;
4. history-only forcing and memory studies plateau around 20.5--20.9 validation
   MAE, identifying future exogenous forcing uncertainty as the main limitation;
5. on the frozen UCI Beijing external test, the no-lag operator has lower MAE and
   sMAPE but higher RMSE than the matched direct model; temporal intervals still
   include zero, so this is suggestive rather than confirmatory evidence.

## Primary checkpoint provenance

The canonical checkpoints are in `checkpoints/tsr_primary/` and contain no delay
input parameters. Their three-seed validation MAE is `16.7630 +/- 0.0252`; their
KnowAir test MAE is `16.1266 +/- 0.0348`. The uniform ensemble reaches
`15.8205 / 24.3253 / 0.3721` for MAE/RMSE/sMAPE. The exact evaluation record is
`artifacts/tsr_primary_test.json`.

The checkpoints were produced by pruning a delay input column that was identically
zero in the selected no-delay models. The pruning changes the parameter count from
72,659 to 72,435 and preserves predictions exactly. The older delayed family is
retained only for the validation ablation.

Both tracks consume realized target-period core meteorology. Comparisons with
AirDDE are comparisons of published point estimates under different information
availability, not identical-input or paired-significance claims.

## Contents

- `RESULTS.json`: canonical numbers and claim constraints.
- `CHECKLIST.md`: manuscript-writing and reporting checks.
- `artifacts/`: compact evidence supporting tables, ablations, temporal
  block-bootstrap analyses, the frozen Beijing replication, the audited GEFSv12
  operational protocol and the unopened China-AQI protocol.
- `checkpoints/`: canonical TSR checkpoints and the retained delay ablation.

All superseded search artifacts, local AirDDE reproduction, broad EDA notebooks,
old China 24-to-6 results and rejected-branch checkpoints were removed during the
paper cleanup.
