# Transport--Source Recurrent Operator

Repo phục vụ manuscript về dự báo PM2.5 nhiều bước trên KnowAir. Research đã được
freeze; kiến trúc paper là recurrent state evolution với wind-aligned transport,
common/local source--sink và không có explicit one-step lag, event expert hay memory
module.

## Kết luận chính

- Recurrent evolution đạt validation MAE `16.7668`, tốt hơn matched direct model
  `18.8620` khoảng `2.0952` MAE.
- Bỏ source/sink làm xấu `+3.6224`; bỏ transport làm xấu `+0.5132`.
- Bỏ explicit lag sau retraining thay đổi `-0.0038`, nên branch này không được giữ.
- Các history-only forcing/memory variants dừng quanh `20.5--20.9` hoặc tệ hơn;
  future exogenous forcing uncertainty là limitation chính.

Các số, claim constraints và artifact dùng để viết paper nằm trong [`paper/`](paper/README.md).

## Cảnh báo provenance

Headline KnowAir test `16.1285 +/- 0.0932` và uniform ensemble
`15.8130 / 24.3084 / 0.3718` đến từ frozen core-meteorology checkpoints **có**
`use_lagged_transport=true`. Kiến trúc no-lag được chọn cuối cùng hiện chỉ có
validation result `16.7630 +/- 0.0252`; không được relabel headline test thành
no-lag result.

Model dùng realized target-period core meteorology. Vì vậy comparison với AirDDE
chỉ là published point estimates dưới information setting khác, không phải fair
identical-input comparison hay paired statistical superiority.

## Setup RTX 5060 Ti

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

## Reproduce

Xem [`RUN_ANALYSIS.md`](RUN_ANALYSIS.md) cho training/ablation validation-only và
China-AQI protocol audit. Repo không còn local AirDDE reproduction và không chứa
command mở corrected China-AQI test.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`data/` và `.venv/` là local resources, không thuộc paper package.
