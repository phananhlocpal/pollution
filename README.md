# KnowAir Quantile Router

Repository này chỉ giữ lại thí nghiệm chọn giữa ba PM trajectories:

- DART q10 (low expert);
- Boundary-CSIO point forecast (center expert);
- DART q90 (high expert).

Oracle chọn expert có station-day MAE thấp nhất và đạt validation MAE
`13.8772215719`. Causal router chỉ dùng thông tin có tại forecast origin và đạt
`20.5219`, nên oracle là ceiling chẩn đoán chứ không phải deployable score.

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

Frozen inputs đã được gom vào
`artifacts/knowair_quantile_router_eda/router_inputs.npz`; không còn phụ thuộc
artifact hoặc code từ các ý tưởng khác. Xem [protocol](docs/QUANTILE_ROUTER.md)
và [retained report](artifacts/knowair_quantile_router_eda/REPORT_VI.md).

## Kiểm thử

```bash
.venv/bin/python -m pytest -q
```
