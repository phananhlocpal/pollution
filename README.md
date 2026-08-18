# KnowAir Quantile Router

Repository này chỉ giữ lại thí nghiệm chọn giữa ba PM trajectories:

- DART q10 (low expert);
- Boundary-CSIO point forecast (center expert);
- DART q90 (high expert).

Oracle hindsight chọn expert có station-day MAE thấp nhất và đạt validation MAE
`13.8772215719`. Causal router chỉ dùng thông tin có tại forecast origin và đạt
`20.5219`. Vì oracle nhìn PM tương lai để quyết định expert, đây là
**hindsight restricted-expert oracle score**, không phải upper bound của causal
forecasting và không phải deployable score.

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Dataset cần nằm tại:

```text
data/benchmarks/knowair/KnowAir.npy
data/benchmarks/knowair/city.txt
```

## Tái lập oracle 13.877

```bash
.venv/bin/python scripts/eda_knowair_quantile_router.py --oracle-only
```

## Chạy lại causal router

```bash
.venv/bin/python scripts/eda_knowair_quantile_router.py \
  --output-dir artifacts/knowair_quantile_router_eda
```

`router_inputs.npz` là local frozen artifact (bị gitignore), nên clean clone
không thể tái lập router đầy đủ nếu không được cấp bundle tương ứng. Báo cáo
tracked chỉ ghi kết quả đã chạy. Xem [protocol](docs/QUANTILE_ROUTER.md),
[retained report](artifacts/knowair_quantile_router_eda/REPORT_VI.md), và
[E001 information-set ladder](docs/METEOROLOGICAL_INFORMATION_LADDER.md).
Lưu ý: report/CSV đang tracked là run lịch sử trước khi thêm purge cho train-tail
tuning; không dùng chúng như kết quả của protocol đã hiệu chỉnh cho đến khi rerun.

## E001: audit information set

E001 chỉ chạy trên validation và kiểm tra archive khí tượng có thực sự được
phát hành tại forecast origin hay không. Nó không báo MAE PM khi chưa có một
model architecture frozen dùng chung cho toàn bộ ladder.

```bash
.venv/bin/python scripts/audit_information_set_ladder.py \
  --forecast-weather data/forecasts/knowair_operational_weather.npz
```

Không truyền `--forecast-weather` sẽ sinh preflight có trạng thái blocked, thay
vì lén dùng realized future weather như forecast vận hành.

## Kiểm thử

```bash
.venv/bin/python -m pytest -q
```
