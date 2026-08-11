# Deep EDA: evidence and research directions

Dataset: UCI Beijing Multi-Site Air Quality, 12 stations, hourly data from
2013-03-01 to 2017-02-28. The chronological split is 70/15/15. All thresholds,
imputation statistics and scaling used by diagnostic probes are fitted on the
training interval only.

This report generates hypotheses. Correlations and diagnostic models are not
causal evidence and do not establish novelty by themselves.

## 1. The forecasting mechanism changes with horizon

Decomposing each station's future PM2.5 change into a city-wide mean change and
a station residual gives the following regional share of component variance:

| Horizon | Regional share |
|---:|---:|
| +1 h | 36.2% |
| +6 h | 67.0% |
| +12 h | 77.4% |
| +24 h | 84.6% |

The graph should therefore not be expected to serve the same purpose at every
horizon. At long horizons, neighbouring stations mostly reveal a common
regional state. At short horizons, local transitions and transport residuals
remain substantial.

## 2. Geography is a strong prior, with station-specific exceptions

Distance has a strong monotonic association with both raw PM2.5 similarity
(Spearman rho = -0.916) and regional-residual change similarity (rho = -0.711).
The mean overlap between geographic four-nearest neighbours and the empirical
top-four residual-change neighbours is 70.8%.

The overlap is only 25% for Changping and Shunyi, and 50% for Dingling and
Gucheng. A global learned graph is not implied. A more testable hypothesis is a
fixed geographic prior plus a small station-specific functional correction,
especially for peripheral stations.

## 3. Wind provides a narrow directional lag signal

Across 48 directed geographic neighbour edges, the strongest unconditional
source-target change association is at lag zero for 97.9% of edges. This is
consistent with common regional forcing, not widespread delayed propagation.

Conditioning on source-station wind changes the picture only at short lags:

| Lag | Mean aligned minus opposed rho | Positive edge fraction |
|---:|---:|---:|
| 0 h | -0.013 | 39.6% |
| +1 h | +0.109 | 91.7% |
| +3 h | +0.035 | 75.0% |
| +6 h | -0.000 | 43.8% |
| +12 h | -0.006 | 39.6% |

At +1 h, the mean contrast is positive in train (+0.107), validation (+0.115)
and test (+0.118), in all four seasons (+0.100 to +0.122), and for moderate
(+0.105) and strong wind (+0.145). This is direct, reproducible evidence for a
short-horizon wind-conditioned directional residual. It is not evidence for
replacing all graph edges with a wind-dynamic graph at every horizon.

## 4. Spikes contain local and regional event types

Among hours with at least one station above its station-specific train p90:

- 27.9% involve only 1–2 stations;
- 17.9% involve 3–5 stations;
- 54.2% involve 6–12 stations.

There are 646 contiguous network spike episodes. Median duration is 3 h, p90 is
30 h, and 29.9% eventually reach at least half of the network. A single spike
MAE merges distinct questions: local onset, regional episode growth and
high-level persistence.

## 5. Spatial information has conditional rather than uniform value

A leakage-safe Ridge probe was used only to locate regimes where spatial
features add information after local history and meteorology. Mean MAE gain
from adding spatial features is +0.54 at +1 h, +1.47 at +6 h, +1.02 at +12 h
and +0.60 at +24 h.

The largest gains concentrate around +6 h, large spatial gaps, high pollution,
winter, and peripheral stations such as Shunyi, Changping and Huairou. At +24 h
the gain is larger in calm than strong wind. This suggests that the long-horizon
spatial branch mainly supplies regional context, while directional wind
transport is a separate short-horizon mechanism.

## Research candidates

1. **Horizon-conditioned regional/local decomposition.** Predict an explicit
   regional factor and station residual, with a horizon-dependent mixture.
2. **Sparse one-hour wind transport residual.** Apply wind alignment only to
   source-to-target residual messages at short lag, on top of the fixed graph.
3. **Regime-gated spatial correction.** Let simple observed regimes such as
   spatial disagreement, pollution level and season control spatial correction.
4. **Dual event formulation.** Predict local/isolated onset and regional episode
   onset separately, alongside concentration regression.

The generic forms of hierarchical graphs, wind-adjusted edges, regime-adaptive
graphs and extreme-event objectives already exist in the literature. The most
defensible candidate contribution is therefore the empirically identified
phase separation: simultaneous regional context dominates longer horizons,
whereas wind-conditioned directional transport is a sparse one-hour residual.
Before any model is changed, the wind claim should next be tested on KnowAir or
AirFormer (which contain meteorology), while the regional/local and event claims
can also be tested on KDD Beijing.

Relevant primary works used to delimit (not prove) novelty:

- [HighAir](https://arxiv.org/abs/2101.04264) already combines hierarchical
  graphs and wind-adjusted edge weights.
- [AirFormer](https://arxiv.org/abs/2211.15979) already separates deterministic
  spatiotemporal representation learning from stochastic uncertainty.
- [PM2.5-GNN](https://arxiv.org/abs/2002.12898) already uses domain knowledge and
  graph learning for fine-grained and long-term PM2.5 effects.
- [DMGENet](https://doi.org/10.1016/j.engappai.2026.115234) already proposes
  horizon- and regime-varying integration of multiple station graphs.
- [TransNet](https://www.nature.com/articles/s44407-026-00052-x) already uses
  real-time meteorology and advection-diffusion-reaction transport operators.

## Reproducible outputs

- `deep_eda_summary.json`: machine-readable summary.
- `regional_local_variance.csv`: horizon decomposition.
- `pairwise_functional_graph.csv`: geographic/functional relation diagnostics.
- `wind_conditioned_pair_lag.csv`: pair-wise lag and wind-regime correlations.
- `wind_alignment_robustness.csv`: split/season/speed robustness.
- `spike_episodes.csv`: event catalogue.
- `diagnostic_ridge_ablation.csv` and `spatial_gain_by_regime.csv`: conditional
  value probes.
- `deep_eda_main.png`: compact visual summary.
