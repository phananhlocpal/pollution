# Quantile Router / Oracle Selector protocol

## Scope được giữ lại

Một station-day selector chọn giữa ba frozen PM trajectories:

1. low expert: DART q10;
2. center expert: Boundary-CSIO;
3. high expert: DART q90.

Validation dùng chronological KnowAir split và không đọc test. Frozen tensor
được đóng gói trong `artifacts/knowair_quantile_router_eda/router_inputs.npz`.

## Oracle selector

Với mỗi forecast origin, future day và station, oracle tính MAE trên tám lead
3-hour của từng expert rồi chọn expert có MAE thấp nhất. Oracle dùng target
tương lai để chọn, vì vậy chỉ là upper bound, không deploy được.

| future day | oracle station-day MAE |
|---:|---:|
| 1 | 12.5281 |
| 2 | 14.3621 |
| 3 | 14.7415 |
| all 72h | **13.8772** |

Lệnh tái lập:

```bash
.venv/bin/python scripts/eda_knowair_quantile_router.py --oracle-only
```

## Causal router

Causal router dùng PM history, frozen predictive spread/regime probability,
predicted event-arrival probability, coordinates và calendar. Hyperparameters,
hard/soft routing, confidence threshold và exact-center fallback được chọn trên
chronological train tail.

Validation outcome:

- center: `20.0813`;
- causal router: `20.5219`;
- oracle selector: `13.8772`.

Kết luận: expert set có oracle ceiling rất mạnh, nhưng branch identity chưa dự
báo được từ feature tại origin. Không được trình bày `13.8772` như causal model
performance.

## Retained files

- `scripts/eda_knowair_quantile_router.py`: implementation;
- `src/shift_pm/data.py`: minimal KnowAir loader;
- `tests/test_quantile_router.py`: selector/router tests;
- `tests/test_data_protocol.py`: split and leakage tests;
- `artifacts/knowair_quantile_router_eda/`: frozen inputs and outputs.
