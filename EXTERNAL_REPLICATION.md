# KnowAir and China-AQI comparison protocol

The project now compares only KnowAir and China-AQI. Architecture selection uses
KnowAir train/validation. China-AQI evaluation uses the AirDDE 96h-to-24h task;
published AirDDE reference metrics are recorded in
`artifacts/external_replication/protocols.json`.

## Current data status

The pinned public AirDDE repository contains KnowAir only. AirDDE attributes its
two external datasets to GAGNN. The official GAGNN repository and its 993 MB
Google Drive archive are now downloaded locally at commit
`509ac7d6eb55914979fc45f6d23e967021cfd270`. The archive is the exact 209-city
China-AQI release: overlapping 24-history/6-target hourly windows, already
separated into train/validation/test. AirDDE's Table 1 also says 209, while one prose
sentence says 203; the released tensor and original GAGNN paper resolve the
station count to 209.

Recreate and verify that checkout/archive with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_gagnn.py
.\.venv\Scripts\python.exe scripts\audit_gagnn_reconstruction.py
```

The audit proves exact unit-stride overlap for `x` and `y`, and exact agreement
between each released target and the target channel 24 hours later. It therefore
reconstructs 96h-to-24h samples independently inside each split, never across a
boundary. Counts are 14,169 train, 1,947 validation and 3,984 test windows.

## Historical China-AQI 24h-to-6h result

After validation training, the three checkpoint hashes were recorded in
`frozen/china_aqi/MANIFEST.json` before the test loader was invoked. The fresh
single-model three-seed test mean is MAE `11.4299 +/- 0.0115`, RMSE
`20.9490 +/- 0.0200`, and MAPE `17.9447 +/- 0.0519`. All three individual MAEs
are between 11.41 and 11.44.
The secondary uniform ensemble reaches `11.3025 / 20.7871 / 17.7295`.

These numbers answer the original GAGNN 24h-to-6h task. They are **not directly
comparable** with AirDDE's published China-AQI `17.03 / 29.91 / 30.82`, which uses
96h history and a 24h forecast.

## Frozen input contract

For an exact dataset not already supported by a direct release adapter, create an
NPZ with:

- `target`: float array `[time, station]`;
- `weather`: float array `[time, station, weather_feature]`;
- `coordinates`: float array `[station, 2]` in longitude/latitude order;
- `station_ids`: optional string array `[station]`.
- `cadence_hours`: optional scalar integer (defaults to one hour).

The weather tensor must preserve all seven published meteorological factors.
Before packing, encode angular wind direction as sine and cosine. The frozen
operator expects wind speed, wind-direction sine and wind-direction cosine at
weather columns 3, 4 and 5. Record the complete raw-to-packed mapping in the
dataset manifest; do not infer it silently.

The direct GAGNN adapter instead preserves its published two-component wind
encoding and reorders released columns to put speed/direction at those same operator
positions; the mapping is recorded in `artifacts/external_replication/protocols.json`.

The loader fits normalization on the first 70% only and exposes chronological
70%/10%/20% train/validation/test partitions. Training is available through:

```powershell
.\.venv\Scripts\python.exe -m common_local.train_dynamics `
  --gagnn-dir data\benchmarks\china_aqi_gagnn `
  --gagnn-protocol 96x24 --future-weather-mode latent `
  --seeds 42 43 44 --epochs 30 --patience 6 --batch-size 64 `
  --disable-lagged-transport --disable-auxiliary --disable-month `
  --output-dir artifacts\china_aqi_96x24_latent_v2 `
  --device cuda
```

The GAGNN release exposes historical covariates, not realized future covariates, so
the latent-forcing mode is mandatory for the corrected comparison. It evolves
global/local meteorological latent states without reconstructing target-period
weather, and the V2 transport operator has no explicit lag branch. Freeze the
three checkpoints with `scripts/freeze_china_aqi_96x24.py`, then and only then run
`scripts/evaluate_china_aqi_96x24.py --allow-test`. The training command itself
remains validation-only.

## V2 selection outcome

The three-seed history-only V2 completed on KnowAir validation with MAEs
`20.6533 / 20.8487 / 20.8613`, mean `20.7878 +/- 0.0952`. This failed the
predeclared `<=18.0` gate. V2 is therefore rejected, no corrected China-AQI
checkpoint was frozen, and its 96h-to-24h test remains unopened. The reconstructable
China protocol is ready, but running it with this rejected architecture would be
test-driven model development rather than a valid external confirmation.

## Reproducibility gate

Before accepting a result, verify exact station count (209), raw feature
names/order, time range and cadence, sample counts after windowing, authors'
missing-data treatment, target definition/unit, and that the split boundaries
match the published benchmark. Set `--history` and `--horizon` from the authors'
exact protocol rather than assuming KnowAir's 24/24 setting. If any item is unknown, label the run as a new
dataset experiment rather than an AirDDE benchmark replication.
