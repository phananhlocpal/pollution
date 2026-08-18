# KnowAir causal PM-distribution router

Đây là artifact duy nhất được giữ lại trong repository. Frozen low/center/high
experts, targets và causal-router covariates đã được gom vào
`router_inputs.npz`; không còn phụ thuộc artifact của các thử nghiệm khác.

Tái lập riêng oracle ceiling:

```bash
.venv/bin/python scripts/eda_knowair_quantile_router.py --oracle-only
```

Router chỉ chọn giữa DART q10, Boundary-CSIO center và DART q90. Feature gồm
PM history, frozen predictive spread/regime probability và arrival probability
đã cross-fit từ weather history. Hyperparameter, hard/soft rule, confidence và
exact fallback đều chọn trên chronological train tail. Test không được đọc.

## Train-selected protection

| future_day | max_leaf_nodes | min_samples_leaf | rule | confidence | train_tail_mae | train_tail_center_mae | active |
|---|---|---|---|---|---|---|---|
| 1.0000 | 7.0000 | 100.0000 | soft | 0.6000 | 16.5941 | 16.6009 | True |
| 2.0000 | 7.0000 | 300.0000 | soft | 0.6000 | 20.7051 | 20.7259 | True |
| 3.0000 | 7.0000 | 100.0000 | soft | 0.4000 | 22.4262 | 22.5602 | True |

## Validation

| future_day | center_mae | routed_mae | oracle_station_day_mae | router_accuracy | center_class_prevalence | low_class_prevalence | high_class_prevalence | active_from_train_gate | n |
|---|---|---|---|---|---|---|---|---|---|
| 1.0000 | 15.9475 | 15.9751 | 12.5281 | 0.5537 | 0.5507 | 0.2783 | 0.1710 | True | 4229915 |
| 2.0000 | 21.1733 | 21.2430 | 14.3621 | 0.4317 | 0.4262 | 0.3580 | 0.2158 | True | 4229923 |
| 3.0000 | 23.1231 | 24.3476 | 14.7415 | 0.3610 | 0.3807 | 0.3624 | 0.2569 | True | 4229936 |

Overall center `20.081`, routed `20.522`, oracle
station-day `13.877`. Khoảng cách lớn giữa
router và oracle đo trực tiếp phần branch identity còn không dự báo được.

**FAIL — oracle quantile ceiling không chuyển thành out-of-time gain.**
