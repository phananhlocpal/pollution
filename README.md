# Common–Local Pollution Forecasting

Repository này giữ một mô hình duy nhất: `common_local` trên benchmark KnowAir.
Các nhánh graph, transport, spatial modes và correction nghiên cứu sau đó đã
được loại bỏ để source phản ánh đúng mô hình đang được giữ làm best model.

## Kiến trúc

Đầu vào là 72 giờ lịch sử (24 bước, cadence 3 giờ) của PM2.5 và khí tượng tại
184 trạm. Mô hình dự báo trực tiếp 72 giờ tiếp theo:

\[
Y_{t+h,n}=C_{t+h}+R_{t+h,n},
\qquad \sum_n R_{t+h,n}=0.
\]

- `common_gru` mã hóa PM2.5 trung bình toàn vùng và khí tượng trung bình.
- `local_gru` mã hóa residual của từng trạm cùng khí tượng tại trạm.
- Hai head dùng future meteorology và horizon embedding.
- Hướng gió được mã hóa bằng `sin/cos`.
- Objective là masked compound L1 cho total/common/residual/increment với trọng
  số `1.0 / 0.25 / 0.25 / 0.10`.

Mô hình có 27,730 parameters. Split KnowAir là chronological `2:1:1`; training
và lựa chọn checkpoint chỉ dùng train/validation. Kết quả được giữ hiện tại là
validation MAE trung bình ba seed `18.6587`.

## Chạy

### Môi trường RTX 50-series

RTX 5060 Ti là GPU Blackwell (`sm_120`) và cần PyTorch build với CUDA 12.8 trở
lên. Môi trường tái lập của repo dùng Python 3.11 và PyTorch 2.8.0/cu128:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-airdde.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

Huấn luyện ba seed:

```bash
python -m common_local.train \
  --epochs 20 --patience 5 --batch-size 8 --device auto
```

Đánh giá lại các checkpoint đã giữ trên validation:

```bash
python -m common_local.train --evaluate-only --batch-size 8 --device auto
```

Output nằm tại `artifacts/common_local/`. Test set không được truy cập trong
pipeline này.

Checkpoint là artifact sinh cục bộ và bị loại khỏi Git bằng `*.pt`. Sau khi train,
export prediction/truth trên validation để chạy evaluator và residual probes:

```powershell
.\.venv\Scripts\python.exe -m common_local.export_predictions
.\.venv\Scripts\python.exe -m benchmarking.evaluator artifacts\predictions\common_local_seed42_val
.\.venv\Scripts\python.exe -m benchmarking.residual_probe artifacts\predictions\common_local_seed42_val
```

Baseline AirDDE được pin ở commit public thông qua Git submodule; xem
`AIRDDE_REPRODUCTION.md` để chuẩn bị data, train và import output vào cùng evaluator.

## Kết quả residual-driven cuối cùng

Sau residual probe và ablation chỉ trên validation, cấu hình được freeze là
`common_local + wind-aligned lag correction + PBL/ventilation`. Baseline được đóng
băng; correction chỉ có 257 tham số trainable (27.987 tham số tổng cộng).

- Validation MAE ba seed: `18.6543 -> 18.5116`.
- Test MAE ba seed: `17.3090 -> 17.2082`.
- AirDDE-repro một seed: validation `16.0741`, test `15.3501`.
- Paired block-bootstrap correction vs baseline trên test: ΔMAE `-0.0867`, CI95%
  `[-0.1007, -0.0719]`.

Regional correction bị loại do gain validation không đáng kể; event head được hoãn
trước khi mở test vì chưa có bằng chứng cho objective MAE. Báo cáo đầy đủ nằm tại
`artifacts/final_analysis/REPORT.md`; manifest freeze nằm tại
`frozen/wind_meteo/MANIFEST.json`.

## Cấu trúc còn lại

- `src/common_local/`: data, model, objective, metrics và training entrypoint.
- `artifacts/common_local/`: checkpoint của seeds 42/43/44 và summary.
- `notebooks/`: năm notebook EDA benchmark.
- `src/benchmark_eda/` và `scripts/`: code tái tạo EDA/download dữ liệu.
- `data/`: datasets và metadata, không bị thay đổi trong quá trình clean.

Chạy kiểm thử:

```bash
python -m pytest -q
```
