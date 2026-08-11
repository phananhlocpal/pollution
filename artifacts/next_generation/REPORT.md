# Next-generation validation analysis

All architecture selection in this report used the training and validation splits.
The exact checkpoints and validation-fitted ensemble weights were recorded in
`frozen/transport_source_recurrent/MANIFEST.json` before the KnowAir test was opened.
The extended models use
PBL, ventilation, dewpoint deficit, and month in addition to the release covariates.
The track previously called strict is renamed **core meteorology with realized
future forcing**: it removes those additional fields but still reads target-period
weather, so exact input-information parity with the AirDDE manuscript is unverified.

## Main result

| Model | Seeds | Parameters | Day 1 MAE | Day 2 MAE | Day 3 MAE | Overall MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| common_local | 3 | 27,730 | 15.8739 | 19.5017 | 20.5874 | 18.6543 |
| static wind + meteo correction | 3 | 27,987 | 15.6521 | 19.3939 | 20.4888 | 18.5116 |
| separated transport/source correction | 3 | 28,212 | 15.6888 | 19.3753 | 20.4542 | 18.5061 |
| Transport--Source Recurrent Operator | 3 | 72,659 | **14.6189** | **17.1793** | **17.8104** | **16.5362 ± 0.0901** |
| recurrent convex prediction ensemble | 3 | 3 x 72,659 | **14.3329** | **16.7215** | **17.3164** | **16.1236** |
| core-meteo future-forcing recurrent | 3 | 72,659 | 14.9403 | 17.4357 | 17.9244 | **16.7668 ± 0.0238** |
| core-meteo future-forcing convex ensemble | 3 | 3 x 72,659 | 14.7156 | 17.1010 | 17.5609 | **16.4592** |
| AirDDE release reproduction | 1 | 360,182 | 14.3537 | 16.6266 | 17.2421 | 16.0741 |

The recurrent operator improves the three-seed mean over static wind+meteo by
1.9754 MAE. The improvement grows with horizon: 1.0332 on Day 1, 2.2146 on Day 2,
and 2.6784 on Day 3. This directly supports the audit's diagnosis that missing
future state evolution, rather than another static feature correction, was the
dominant failure mode.

The validation-fitted convex recurrent ensemble has weights
`[0.2382, 0.3649, 0.3969]`. Against the one-seed AirDDE release validation bundle,
its paired delta is +0.0491 MAE with moving-block CI95% `[-0.1276, 0.2536]`.
The interval remains crossing zero at block lengths 48 (`[-0.1465, 0.3016]`)
and 96 (`[-0.1909, 0.3295]`).
This is not a strict feature-parity or multi-seed AirDDE claim, but the two results
are not distinguishable under the current conditional block bootstrap.

## Cheap experiments and rejected branches

- The static three-seed convex ensemble reached 18.2482 validation MAE. On the
  already-open legacy test it reached 16.9487, 0.0287 above the paper reference.
- Separating zero-mean wind transport from unconstrained source/sink improved all
  three baseline seeds, but only moved the mean static validation result from
  18.5116 to 18.5061. The factorization is sound; static application is insufficient.
- Capacity search up to 93,314 parameters was negative. The best direct-head model
  was hidden 96 at 18.8879; two GRU layers and Huber were also worse. Capacity alone
  does not explain the gain from sequential dynamics.
- A warm-started regime/event expert reached 16.5215 versus its recurrent parent at
  16.4967 and was rejected by global validation MAE.
- Thirty-two SVD modes explain 68.9% of training residual variance, but a causal
  lag-24/48 mode forecast worsened held-out 72h MAE by 1.4380. The learned regional
  mode branch is rejected.

Frozen-checkpoint mechanism knockouts support the operator factorization: removing
transport worsens validation MAE by 5.0102, removing source/sink by 5.8021, removing
lagged transport by 0.7022, and removing month by 0.1878. These are necessity
diagnostics for the learned checkpoint, not substitutes for retraining. The strict
model is the separately retrained fair-input experiment.

The publication-grade from-scratch, three-seed ablation gives a more qualified
mechanism result:

| Retrained core-meteo variant | Parameters | Validation MAE | Delta vs full |
| --- | ---: | ---: | ---: |
| full recurrent | 72,659 | 16.7668 +/- 0.0238 | 0.0000 |
| no transport | 72,659 | 17.2800 +/- 0.0238 | +0.5132 |
| no source/sink | 72,659 | 20.3892 +/- 0.0252 | +3.6224 |
| no lagged transport | 72,659 | 16.7630 +/- 0.0252 | -0.0038 |
| matched direct/non-recurrent | 72,630 | 18.8620 +/- 0.0457 | +2.0952 |

Thus sequential evolution and especially source/sink modeling survive retraining;
transport has a smaller but consistent value. The explicit lagged pathway is not
supported after retraining and should not receive a positive mechanism claim. The
matched direct result rules out parameter count as the explanation for the recurrent
gain. Full seed/day tables are in `artifacts/retrained_ablation/summary.json`.

## Efficiency

On the RTX 5060 Ti, batch 8, 10 warmups and 100 timed forwards:

| Model | Params | Median ms/origin | p95 ms/origin |
| --- | ---: | ---: | ---: |
| common_local | 27,730 | 0.185 | 0.221 |
| separated transport/source correction | 28,212 | 0.265 | 0.296 |
| Transport--Source Recurrent Operator | 72,659 | 4.025 | 6.781 |
| AirDDE release | 360,182 | 98.230 | 119.204 |

The recurrent operator is about 24.4x faster by median forward latency and uses
about 5x fewer parameters than AirDDE release. It gives up the sub-millisecond
latency of direct heads, but retains a large efficiency advantage over the neural
differential-equation implementation.

## Frozen test confirmation against the paper

All three individual recurrent seeds beat the published AirDDE KnowAir MAE of
16.92: seed 42/43/44 obtained 16.4576, 16.4326, and 16.3916 respectively.

The validation-frozen convex ensemble obtained:

| Metric | Ours | AirDDE paper | Difference |
| --- | ---: | ---: | ---: |
| MAE | **16.0537** | 16.92 | **-0.8663 (-5.12%)** |
| RMSE | **24.1650** | 27.78 | **-3.6150** |
| sMAPE | **0.3778** | 0.38 | **-0.0022** |

Its Day 1/2/3 MAE is 13.5869 / 16.7822 / 17.7919. The observed result therefore
beats the paper point estimate overall and at every reported aggregate metric.
Because the paper publishes only aggregate scalars, a paired significance test
against paper predictions is impossible. One-sample temporal-block intervals for
our origin losses widen across block lengths and include 16.92; the claim is about
the frozen benchmark point estimate, not paired statistical superiority.

The separately trained core-meteorology model disables PBL, ventilation, dewpoint
deficit, and month. Its three individual test MAEs are 16.2584 / 16.0441 / 16.0830;
the publication headline is therefore the single-model mean **16.1285 +/- 0.0932**.
All three seeds beat the paper's 16.92 point estimate. The secondary, simpler
uniform three-seed ensemble is reported below; the validation-fitted convex result
(15.8086 MAE) is retained only as a sensitivity result because it improves the
uniform mean by just 0.0043.

| Core-meteo future-forcing metric | Uniform ensemble | AirDDE paper | Difference |
| --- | ---: | ---: | ---: |
| MAE | **15.8130** | 16.92 | **-1.1070 (-6.54%)** |
| RMSE | **24.3084** | 27.78 | **-3.4716** |
| sMAPE | **0.3718** | 0.38 | **-0.0082** |

Its Day 1/2/3 MAE is 13.6984 / 16.5115 / 17.2290. This removes the extra-derived-field
caveat, but not the realized-future-weather caveat. The project-level KnowAir test was
already known from the extended track, so this is not presented as a fresh external
lockbox; strict checkpoint hashes and ensemble weights were fixed before their test
predictions were exported.

## History-only fairness experiment

Two causal seed-43 screening variants were trained and selected on validation only. Repeating the
last observed weather reached 20.7945 MAE; a jointly trained per-station causal GRU
weather forecaster reached 20.7071. Neither forward pass changes when the supplied
future-weather tensor is perturbed, which is covered by a regression test. Both are
rejected on validation and the KnowAir test remains unopened for these variants.
This negative result confirms that realized target-period meteorology accounts for
a material part of the core-meteo track's advantage; it does not support an
identical-information AirDDE claim.

## Residuals before and after recurrence

Across the three validation seeds, recurrence reduces horizon MAE increasingly from
0.051 at 3h to 1.470 at 24h and 2.638 at 72h. At 72h, event-onset MAE falls from
71.53 to 55.82, decay from 27.93 to 20.93, and winter (DJF) from 34.13 to 26.84.
The aligned-wind 3h correction signal changes from -0.126 incremental MAE to +0.022,
and its 24h signal shrinks from -0.058 to -0.005: the recurrent transport pathway
has absorbed most of that residual structure. Remaining bottlenecks are event onset,
winter pollution, and weaker 72h boundary-layer/ventilation signals (-0.0246 and
-0.0147 incremental MAE). Full tables are in
`artifacts/residual_generation_comparison/summary.json`.

## External lockboxes

The frozen operator now has both a dataset-neutral NPZ adapter and a direct adapter
for the official GAGNN release. The 209-city China-AQI archive has been downloaded
and its exact 24h-to-6h, hourly, 7:1:2 protocol is used without future covariate
leakage. (AirDDE's prose says 203 once, but its Table 1 and the source release use
209.) After the three validation checkpoints were hashed and frozen, the fresh test
was opened once:

| China-AQI test metric | Single-model 3-seed mean | AirDDE paper | Relative reduction |
| --- | ---: | ---: | ---: |
| MAE | **11.4299 +/- 0.0115** | 17.03 | **32.88%** |
| RMSE | **20.9490 +/- 0.0200** | 29.91 | **29.96%** |
| MAPE | **17.9447 +/- 0.0519** | 30.82 | **41.78%** |

All individual seed MAEs (11.4399 / 11.4137 / 11.4360) beat the published point
estimate. The secondary uniform ensemble reaches 11.3025 MAE, 20.7871 RMSE and
17.7295 MAPE. Exact US-PM (175 counties) remains absent from both public
repositories, so no substitute is scored under that benchmark name.
`EXTERNAL_REPLICATION.md` records provenance, input contract, freeze gate and commands.

## Optional fairness work

AirDDE release still has only seed 2024. This does not block comparison with the
published paper point estimates. The multi-seed wrapper and hierarchical
seed/block comparison are implemented, but seeds 42/43/44 are a separate multi-hour
fairness job. The approximate paper-style option is explicitly labelled as Huber/SmoothL1
plus patience 10, not as an exact reconstruction of the paper's Bayesian HPO.

Feature-track definitions are recorded in `artifacts/fairness/feature_tracks.json`.
