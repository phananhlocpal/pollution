# Cross-dataset observations

This report compares five benchmark families: UCI Beijing Multi-Site, KDD Cup
2018 (Beijing and London are reported separately), KnowAir, AirFormer-tiny and
AirQualityBench. It records observations and exceptions; it does not select a
model or research direction.

All horizon statistics use future PM2.5 change, station-specific training
scales and chronological splits. Correlations are descriptive. In particular,
`future - current` is algebraically coupled with current level, so the negative
current-versus-change correlation must not be interpreted as causal mean
reversion. Spatial-gap results are also reported after partialling out ranked
current PM to reduce the analogous shared-current artifact.

## Repeated observations

### 1. Persistence weakens continuously with horizon

At the first available horizon, current/future PM2.5 Spearman correlation is
0.88–0.96 across all six reported panels. At +24 h it falls to 0.16–0.58. Median
absolute persistence error, divided by station training IQR, increases from
0.06–0.22 at the native horizon to 0.38–0.84 at +24 h.

This pattern is present in every benchmark, including the small AirFormer
release. Comparing models only at one horizon can therefore conceal a major
change in problem difficulty.

### 2. High current PM tends to be followed by a negative residual change

Current-level versus future-change rho is negative in every dataset and becomes
more negative at longer horizons. At +24 h it ranges from -0.35 to -0.67. The
sign remains negative in every train/validation/test split and every season of
the five long panels.

This is a stable predictive pattern, but part of it is regression-to-the-mean
and mathematical coupling because current PM appears with a negative sign in
the target. It is not by itself evidence for a physical removal mechanism.

### 3. Neighbour trend is consistently more informative than own recent trend

At +6 h and +24 h, neighbour-trend rho exceeds own-trend rho in all 70 tested
dataset × horizon × split/season combinations (five long panels × two horizons
× seven strata). The absolute neighbour correlation can still be small or
negative; the repeated observation is its *relative* advantage over own trend.

This holds for local geographic neighbours in UCI Beijing, KDD Beijing,
KnowAir and AirQualityBench. London uses other-city stations because the TSF
release does not include London coordinates.

### 4. Spatial disagreement contains information beyond current PM

The raw spatial gap `neighbour - own` shares `-own` with the forecast change,
so raw correlation overstates its evidence. After partialling out current PM,
the median station association at +6 h remains positive in all long datasets:

| Panel | Partial rho at +6 h | Partial rho at +24 h |
|---|---:|---:|
| UCI Beijing | 0.181 | 0.074 |
| KDD Beijing | 0.219 | 0.082 |
| KDD London | 0.207 | 0.070 |
| KnowAir | 0.241 | 0.222 |
| AirQualityBench | 0.203 | 0.190 |

Across split and season strata, the +6 h partial association is always
positive. At +24 h it remains consistently positive in KnowAir,
AirQualityBench and UCI Beijing, but becomes weak and occasionally negative in
London. Thus short/intermediate-horizon spatial correction is more universal
than its 24-hour strength.

### 5. Most inter-node PM change is synchronous rather than delayed

For geographic/context edges, lag zero is the strongest tested change
association for 93.5–97.9% of edges in UCI Beijing, KDD Beijing, KDD London,
KnowAir and AirQualityBench. Median lag-zero rho ranges from 0.13 to 0.40 and
falls close to zero by +6 h or +12 h.

This repeated result distinguishes broad synchronous variation from a smaller
directed transport residual. AirFormer-tiny is excluded because its 60 windows
cannot be joined into a chronological lead-lag series.

### 6. Geographic distance predicts both level and transition similarity

Distance has a negative Spearman association with pairwise PM level correlation
in every dataset with coordinates: -0.916 UCI Beijing, -0.865 KDD Beijing,
-0.723 KnowAir and -0.810 AirQualityBench. It also predicts similarity of
native-step PM changes: -0.888, -0.825, -0.448 and -0.796 respectively.

The relationship weakens, but does not disappear, when moving from city scale
to nationwide scale in KnowAir.

### 7. A delayed wind-alignment contrast replicates in two datasets

In UCI Beijing, aligned-minus-opposed edge correlation is approximately zero at
lag zero (-0.013), rises to +0.109 at +1 h, and returns to approximately zero at
+6 h and +12 h. In KnowAir, it is also near zero at lag zero (-0.006), rises to
+0.066 at +3 h and +6 h, then declines to +0.009 at +12 h.

The first resolvable delayed contrast is positive across train, validation,
test, all four seasons and moderate/strong wind in both datasets. Positive-edge
fractions are 91.7% in UCI at +1 h and 76.3% in KnowAir at +3 h. The exact lag
cannot be directly equated because KnowAir is sampled every three hours.

### 8. Extreme hours are strongly locally clustered

Using each station's training p90, the observed fraction of neighbours also in
an extreme state when a target station is extreme is 0.45–0.85. Under temporal
independence based on each neighbour's marginal test rate, the expected value
is only 0.04–0.14. The resulting local event lift is 5.5–10.5 across all five
long panels.

This is not merely a Beijing phenomenon: the largest lift occurs in nationwide
KnowAir. At the same time, active-node fractions differ greatly with spatial
scope, so a “regional spike” must be defined relative to the dataset's spatial
domain.

### 9. Distance to the station-specific threshold strongly orders onset risk

Across UCI Beijing, both KDD cities, KnowAir and AirQualityBench, 24-hour onset
probability increases monotonically over all four current/P90 bins. In the
0.75–1.00 bin it is 55.3–66.5%; in the lowest bin it is 2.7–15.5%.

AirQualityBench is the exception in magnitude: even its lowest bin has a 15.5%
onset rate, consistent with its heavier tail, greater heterogeneity and possible
cross-provider/reporting effects. AirFormer-tiny cannot support a population
onset rate.

### 10. Extreme events have short medians and long duration tails

Median station-level extreme episode duration is 2–6 h, while p90 duration is
10–28 h. The tail appears in every long dataset. Consequently, an extreme
observation may represent a new onset, continued episode or decay phase; those
states are observationally different even at the same PM level.

### 11. PM distributions are heavy-tailed everywhere

The PM2.5 p99/p90 ratio ranges from 1.92 to 3.02 across all releases. The global
AirQualityBench panel has the heaviest relative tail. Native-step downward p99
magnitudes are larger than upward p99 magnitudes in UCI Beijing, both KDD
cities and KnowAir; AirQualityBench is approximately symmetric. This asymmetry
is descriptive and may combine atmospheric decay, clipping and data-provider
effects.

### 12. Multi-pollutant co-movement is clearer than precursor value

Across UCI Beijing, KDD Beijing and AirQualityBench:

- PM10 change has the strongest positive contemporaneous coupling with PM2.5
  change (median station rho 0.49–0.71).
- CO and NO2 changes are positively coupled (0.30–0.50 and 0.33–0.39).
- O3 change is negatively coupled (-0.20 to -0.12).
- After controlling current PM2.5, current co-pollutant level has weak and
  inconsistent association with future PM2.5 change.

Thus co-pollutants robustly describe the current pollution regime, while their
standalone leading information is much less transferable.

### 13. Some meteorological signs transfer; directional components do not

Across UCI Beijing and KnowAir, after controlling current PM:

- wind speed and precipitation have negative association with +6 h and +24 h
  PM2.5 change;
- pressure has positive association at +24 h;
- temperature and dew point are negative at +24 h, much more strongly in
  KnowAir;
- east/north wind-component signs differ by dataset and location.

Scalar ventilation/washout indicators therefore show more cross-dataset sign
consistency than raw directional components. Direction becomes more comparable
only when projected onto a source-target edge, as in the wind-alignment test.

### 14. Season and hour-of-day dominate weekday/weekend effects

Seasonal median amplitude is 0.27–1.04 training-IQR units, and diurnal amplitude
is 0.15–0.31. In contrast, weekend-minus-weekday median is between -0.01 and
+0.02 IQR in every long panel. Calendar information is therefore not equally
valuable: month/season and hour show visible structure, while a global weekend
indicator is weak.

### 15. Missingness is a benchmark property, not a universal data property

Mean PM2.5 coverage is 97.9% in UCI, 86.8% in KDD Beijing, 80.5% in KDD London
and 77.3% even among the 250 highest-coverage AirQualityBench stations. KnowAir
and AirFormer-tiny contain no nonfinite values in their released tensors,
indicating preprocessing rather than demonstrating that real sensors have no
gaps.

Maximum observed gap length ranges from 343 h in UCI to 8,106 h in KDD London.
Long gaps coexist with strong persistence and can distort both temporal and
spatial correlation if filled without preserving a mask.

### 16. Spatial scope changes the apparent common factor

At +24 h, the cross-node mean component accounts for 76–85% of component
variance in city-scale UCI/KDD panels, but only 3–19% in nationwide/global
KnowAir, AirFormer-tiny and AirQualityBench. This statistic is intentionally not
a universal physical parameter: averaging a city, a country and the globe
defines different factors.

The contrast is itself an observation: conclusions about “regional dominance”
are conditional on the geographic support of the dataset.

### 17. Chronological distribution shift appears in every long release

Test-minus-train median PM2.5 ranges from -0.25 to +0.06 training-IQR units in
the long panels; AirFormer-tiny differs by +0.15 IQR between packaged train and
test targets. The direction is not universal, but equality of train and test
distributions is unsupported everywhere.

### 18. Between-node heterogeneity grows with geographic/provider scale

The standard deviation of node medians is only 0.07–0.23 global-IQR units for
the city/nation panels, versus 0.49 in AirFormer-tiny and 0.77 in global
AirQualityBench. Part of the latter may reflect climate, emissions, station
type, provider and unit/reporting differences rather than atmospheric spatial
structure alone.

## Important non-common observations

- The city-scale regional component grows strongly with horizon; the same is
  not true of a nationwide/global mean.
- Long-horizon partial spatial-gap association remains strong in KnowAir and
  AirQualityBench but becomes weak in Beijing/London city panels.
- Missingness is absent from the processed KnowAir/AirFormer tensors but severe
  in KDD and AirQualityBench.
- AirQualityBench has a much heavier relative tail, higher low-bin onset rate
  and greater station heterogeneity than the other releases.
- Directional wind evidence is currently comparable only in UCI and KnowAir;
  absence of the test in other releases is missing evidence, not a negative
  result.

## Validity limits

- AirFormer-tiny contains only 20 windows per split. It supports tensor,
  horizon and within-window spatial diagnostics, not event frequencies,
  seasonality, episode duration or chronological lead-lag claims.
- AirQualityBench calculations use the 250 highest-coverage PM2.5 stations to
  keep mask-aware five-year analysis tractable.
- KDD London does not ship coordinates in the TSF package, so its context set is
  not a geographic KNN.
- Correlation, partial correlation and temporal ordering do not establish
  causality or source attribution.
