# Transport--Source Recurrent Operator

Repo phục vụ manuscript về dự báo PM2.5 nhiều bước trên KnowAir. Research đã được
freeze; kiến trúc paper là recurrent state evolution với wind-aligned transport,
common/local source--sink và không có explicit one-step lag, event expert hay memory
module.

## Kết luận chính

- TSR đạt validation MAE `16.7630`, tốt hơn matched direct model `18.8620`
  khoảng `2.0990` MAE.
- Trong ma trận ứng viên có trễ, bỏ source/sink làm xấu `+3.6224`; bỏ transport
  làm xấu `+0.5132`.
- Thêm explicit lag vào TSR làm xấu `+0.0038`; block-bootstrap không cho thấy
  lợi ích, nên kiến trúc chính không chứa nhánh này.
- Trên KnowAir test, ba mô hình TSR đạt `16.1266 +/- 0.0348`; uniform ensemble
  đạt `15.8205 / 24.3253 / 0.3721` cho MAE/RMSE/sMAPE.
- Các history-only forcing/memory variants dừng quanh `20.5--20.9` hoặc tệ hơn;
  future exogenous forcing uncertainty là limitation chính.
- Trên UCI Beijing độc lập, TSR giảm point estimate MAE và sMAPE nhưng tăng RMSE;
  khoảng tin cậy thời gian vẫn cắt 0 nên kết quả chỉ mang tính gợi ý.

Các số, claim constraints và artifact dùng để viết paper nằm trong [`paper/`](paper/README.md).

## Ranh giới so sánh

Checkpoint chính nằm trong `paper/checkpoints/tsr_primary/` và không có đầu vào
trễ. Chúng được chọn trên validation trước khi đánh giá KnowAir test; kết quả đầy
đủ nằm trong `paper/artifacts/tsr_primary_test.json`.

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
