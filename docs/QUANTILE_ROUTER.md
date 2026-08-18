# Quantile Router / Oracle Selector protocol

## Scope được giữ lại

Một station-day selector chọn giữa ba frozen PM trajectories:

1. low expert: DART q10;
2. center expert: Boundary-CSIO;
3. high expert: DART q90.

Validation dùng chronological KnowAir split và không đọc test. Frozen tensor
`artifacts/knowair_quantile_router_eda/router_inputs.npz` là local artifact bị
gitignore; nó cần được phân phối cùng manifest/hash nếu muốn tái lập độc lập.

## Oracle selector

Với mỗi forecast origin, future day và station, oracle tính MAE trên tám lead
3-hour của từng expert rồi chọn expert có MAE thấp nhất. Oracle dùng target
tương lai để chọn, vì vậy không deploy được. Thuật ngữ đúng là **hindsight
restricted-expert oracle**, không phải upper bound cho mọi mô hình causal có
thể có.

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

`MANIFEST.json` ghi hash của bundle local đã dùng để tái lập score này.

## Causal router

Causal router dùng PM history, frozen predictive spread/regime probability,
predicted event-arrival probability, coordinates và calendar. Hyperparameters,
hard/soft routing, confidence threshold và exact-center fallback được chọn trên
chronological train tail. Fitting labels và train-tail targets được purge 24
steps (72h), để các target window không overlap.

Validation outcome:

- center: `20.0813`;
- causal router: `20.5219`;
- oracle selector: `13.8772`.

Kết luận: expert set có oracle ceiling rất mạnh, nhưng branch identity chưa dự
báo được từ feature tại origin. Không được trình bày `13.8772` như causal model
performance.

## Historical report status

`REPORT_VI.md`, CSV và JSON currently tracked là kết quả trước khi code thêm
purge 24-step cho train-tail tuning. Chúng vẫn là bằng chứng negative cho router
trước đó, nhưng không phải kết quả của protocol đã hiệu chỉnh. Rerun trước khi
trích dẫn lại bất kỳ routed-MAE nào.

## Retained files

- `scripts/eda_knowair_quantile_router.py`: implementation;
- `src/shift_pm/data.py`: minimal KnowAir loader;
- `tests/test_quantile_router.py`: selector/router tests;
- `tests/test_data_protocol.py`: split and leakage tests;
- `artifacts/knowair_quantile_router_eda/`: retained reports/outputs; NPZ
  inputs and predictions are local and not versioned.
