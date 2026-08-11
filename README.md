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
