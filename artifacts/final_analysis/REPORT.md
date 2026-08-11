# Final residual-driven analysis

Architecture freeze: `frozen/wind_meteo/MANIFEST.json`.

## Unified metrics

| model | split | seeds | day1_mae_mean | day2_mae_mean | day3_mae_mean | mae_mean | mae_std | rmse_mean | smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common_local | val | 3 | 15.8739 | 19.5017 | 20.5874 | 18.6543 | 0.0649 | 29.7220 | 0.3961 |
| common_local+wind+meteo | val | 3 | 15.6521 | 19.3939 | 20.4888 | 18.5116 | 0.0501 | 29.5335 | 0.3936 |
| AirDDE-repro | val | 1 | 14.3537 | 16.6266 | 17.2421 | 16.0741 | 0.0000 | 26.3022 | 0.3449 |
| common_local | test | 3 | 14.4927 | 18.1889 | 19.2453 | 17.3090 | 0.1437 | 26.9845 | 0.3996 |
| common_local+wind+meteo | test | 3 | 14.3306 | 18.1162 | 19.1779 | 17.2082 | 0.1440 | 26.8464 | 0.3975 |
| AirDDE-repro | test | 1 | 13.4404 | 16.0018 | 16.6079 | 15.3501 | 0.0000 | 24.1722 | 0.3540 |

## Paired inference

- Selected vs baseline validation: ΔMAE -0.1223, CI95% [-0.13804118223488332, -0.10609256383031607].
- Selected vs baseline test: ΔMAE -0.0867, CI95% [-0.10072277709841729, -0.0719185095280409].
- Selected vs AirDDE test: ΔMAE 1.7785, CI95% [1.401084065437317, 2.1648944437503816].

The AirDDE comparison is one released-code seed (2024). Paper MAE 16.92 is retained only as an external reference.
