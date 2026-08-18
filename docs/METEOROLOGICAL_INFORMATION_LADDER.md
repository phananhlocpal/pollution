# E001 — Meteorological Information-Set Ladder

## Question

Giữ nguyên split, history (72h), horizon (72h), loss, seed và training budget:
skill PM2.5 thay đổi bao nhiêu khi thay đổi riêng information set của khí tượng?

## Rungs

| rung | permitted information at forecast origin | reporting status |
|---|---|---|
| PM history | PM đến thời điểm origin | operational |
| PM + past weather | khí tượng quan sát đến origin | operational |
| persistence / climatology | deterministic transform của past weather | operational baseline |
| archived forecast weather | dự báo có issue time ≤ origin | operational nếu audit pass |
| realized future weather | khí tượng quan sát trong target period | perfect-prognosis diagnostic only |

Mọi rung phải dùng cùng forecast origins, train-only scaler, checkpoint-selection
rule, parameter budget và seeds. Không đọc KnowAir test: dataset này chỉ là
development/diagnostic dataset trong hướng nghiên cứu hiện tại.

## Forecast archive contract

`scripts/audit_information_set_ladder.py` nhận một NPZ không-pickle có:

- `forecast_start[S]`: integer KnowAir origin;
- `issue_time_ns[S]`: UTC nanoseconds khi forecast đã được phát hành;
- `valid_time_ns[S,H]`: UTC nanoseconds cho từng lead;
- `weather[S,H,N,W]`: giá trị weather forecast;
- `weather_feature_names[W]`: channel names trùng raw KnowAir weather names;
- `metadata_json`: scalar JSON gồm `source`, `model_version`, và
  `time_basis: "UTC"`.

Audit reject archive nếu: issue time sau origin; valid time không khớp chính xác
với lead; coverage thiếu validation origins; duplicate origins; số trạm/channel
không hợp lệ; non-finite values; hoặc window vượt validation vào test.

```bash
.venv/bin/python scripts/audit_information_set_ladder.py \
  --forecast-weather data/forecasts/knowair_operational_weather.npz
```

Nếu thiếu archive, command vẫn tạo `e001_preflight.json` với trạng thái
`blocked_missing_archive`. Đây là intentional: persistence và synthetic noise
hữu ích cho diagnostic, nhưng không được gọi là operational forecast weather.

## Decision rule

Chỉ khi archive audit pass mới chạy cùng một frozen PM architecture cho toàn bộ
ladder, báo MAE Day 1/2/3/overall với ba seeds và block-bootstrap theo forecast
origin. Expected pattern của H1 là:

`realized future weather < archived forecast weather < history-only`.

Không có ordering này thì H1 bị yếu đi; không mở rộng architecture trước E001–E003.
