# AirDDE external-lockbox replication

The Transport--Source Recurrent Operator is frozen on KnowAir. China-AQI and
US-PM are treated as new lockboxes: no architecture redesign or test-driven
selection is allowed. Their published AirDDE reference metrics are recorded in
`artifacts/external_replication/protocols.json`.

## Current data status

The pinned public AirDDE repository contains KnowAir only. AirDDE attributes its
two external datasets to GAGNN. The official GAGNN repository and its 993 MB
Google Drive archive are now downloaded locally at commit
`509ac7d6eb55914979fc45f6d23e967021cfd270`. The archive is the exact 209-city
China-AQI release: 24 historical hourly steps, 6 forecast steps, and already
separated 70%/10%/20% windows. AirDDE's Table 1 also says 209, while one prose
sentence says 203; the released tensor and original GAGNN paper resolve the
executable protocol to 209.

Recreate and verify that checkout/archive with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_gagnn.py
```

The corresponding exact 175-county US-PM tensor is not in either public repository
and remains unavailable. Similar US datasets are not substituted under that name.

## Frozen China-AQI result

After validation training, the three checkpoint hashes were recorded in
`frozen/china_aqi/MANIFEST.json` before the test loader was invoked. The fresh
single-model three-seed test mean is MAE `11.4299 +/- 0.0115`, RMSE
`20.9490 +/- 0.0200`, and MAPE `17.9447 +/- 0.0519`, versus published AirDDE
`17.03 / 29.91 / 30.82`. All three individual MAEs are between 11.41 and 11.44.
The secondary uniform ensemble reaches `11.3025 / 20.7871 / 17.7295`.

## Frozen input contract

For an exact dataset not already supported by a direct release adapter, create an
NPZ with:

- `target`: float array `[time, station]` (AQI for China-AQI, PM2.5 for US-PM);
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
  --seeds 42 43 44 --epochs 30 --patience 6 --batch-size 256 `
  --disable-auxiliary --disable-month --future-weather-mode learned `
  --output-dir artifacts\china_aqi_history_learned `
  --device cuda
```

The GAGNN release exposes historical covariates, not realized future covariates, so
the causal learned-weather mode is mandatory here. Freeze the three checkpoints
with `scripts/freeze_china_aqi.py`, then and only then run
`scripts/evaluate_china_aqi.py --allow-test`. For US-PM, use the generic NPZ route
with `--expected-stations 175` once the exact tensor is obtained. The training
command itself remains validation-only.

## Reproducibility gate

Before accepting a result, verify exact station count (209 or 175), raw feature
names/order, time range and cadence, sample counts after windowing, authors'
missing-data treatment, target definition/unit, and that the split boundaries
match the published benchmark. Set `--history` and `--horizon` from the authors'
exact protocol rather than assuming KnowAir's 24/24 setting. If any item is unknown, label the run as a new
dataset experiment rather than an AirDDE benchmark replication.
