# EDA KnowAir: khí tượng 72 giờ quá khứ và 72 giờ tương lai

## Kết luận ngắn

Quy luật chung mạnh nhất là **persistence có thời hạn**: quan sát càng gần mốc dự báo càng liên hệ mạnh với tương lai, nhưng tốc độ mất thông tin khác nhau rõ theo biến. Sau khi loại trung bình khí hậu riêng cho từng `trạm × tháng × giờ theo clock của dataset`, nhiệt độ, áp suất và độ ẩm còn tín hiệu hữu ích trong khoảng 1–2 ngày; gió chỉ còn tín hiệu mạnh trong vài giờ đầu.

Không nên diễn giải tương quan thô là khả năng dự báo. Ví dụ, tương quan nhiệt độ giữa giá trị cuối quá khứ và lead 72h là 0.907 trên dữ liệu thô nhưng chỉ còn 0.212 sau khi bỏ mùa/giờ; áp suất tương ứng giảm từ 0.767 xuống 0.091. Phần lớn “khả năng nhìn xa 72h” trong tương quan thô là khí hậu học, không phải thông tin động lực từ cửa sổ quá khứ.

## Phạm vi và cách tính

- Dùng toàn bộ KnowAir từ `2015-01-01 00:00` đến `2018-12-31 21:00`, không dùng ranh giới train/validation/test.
- Dữ liệu cách nhau 3 giờ, 184 trạm, 11,688 mốc thời gian.
- Mỗi mẫu gồm 24 bước (72h) quá khứ liền với 24 bước (72h) tương lai.
- Quét toàn bộ 11,641 origin hợp lệ. Các cửa sổ được phép chồng lấn đúng theo yêu cầu.
- Tính Pearson `r` riêng ở từng trạm rồi báo cáo median và IQR trên 184 trạm, tránh để vài trạm có phương sai lớn chi phối.
- Phân tích `raw` giữ nguyên mùa/chu kỳ ngày đêm. Phân tích `anomaly` trừ trung bình bốn năm của từng `trạm × tháng × giờ theo clock của dataset`; metadata hiện có không đủ để khẳng định timezone nên báo cáo không gán nhãn UTC/local.
- Hướng gió được xử lý bằng vector `u/v` và cosine/góc lệch; không tính Pearson trực tiếp trên góc 0–360°.

## Persistence từ giá trị khí tượng cuối cùng

Các số dưới đây là median station correlation trên anomaly:

| Biến | +3h | +24h | +48h | +72h | Lead đầu tiên `r < 0.5` |
|---|---:|---:|---:|---:|---:|
| Nhiệt độ 2m | 0.929 | 0.646 | 0.380 | 0.212 | 33h |
| Áp suất bề mặt | 0.985 | 0.618 | 0.257 | 0.091 | 33h |
| RH 950 hPa | 0.931 | 0.510 | 0.251 | 0.148 | 27h |
| Tốc độ gió 100m | 0.817 | 0.205 | 0.034 | -0.012 | 9h |

Nhiệt độ có một “vai” nhẹ khoảng lead 15–24h thay vì giảm đơn điệu hoàn toàn. Đây có thể là nhịp ngày đêm còn lại trong anomaly (ví dụ biên độ hoặc pha thay đổi theo ngày), nhưng hiệu ứng nhỏ hơn rất nhiều so với chu kỳ thấy trong dữ liệu raw.

Với hướng gió, median góc lệch trung bình là 23.1° ở +3h, 64.8° ở +24h, 77.5° ở +48h và 79.4° ở +72h. Vì vậy hướng gió gần nhất có giá trị ngắn hạn, nhưng không phải tín hiệu ổn định cho cả horizon 72h.

## Cả cửa sổ 72h mang thông tin gì?

Trung bình 72h quá khứ so với trung bình 72h tương lai trên anomaly có median `r`:

- nhiệt độ: 0.395 (IQR 0.380–0.427);
- độ ẩm: 0.349 (0.295–0.396);
- áp suất: 0.237 (0.199–0.276);
- tốc độ gió: 0.096 (0.052–0.131).

Ở mọi lead được kiểm tra, giá trị gần origin liên hệ với tương lai mạnh hơn trung bình toàn bộ 72h. Ví dụ tại +24h, `last value` so với `past 72h mean` lần lượt là 0.646 so với 0.391 cho nhiệt độ, 0.618 so với 0.270 cho áp suất, 0.510 so với 0.307 cho RH, và 0.205 so với 0.063 cho tốc độ gió. Quy luật thực dụng là **recency quan trọng hơn average dài hạn**; nếu đưa 72h quá khứ vào model thì nên cho phép trọng số giảm theo tuổi dữ liệu hoặc attention theo lag.

## Quan hệ chéo giữa các biến

Sau khi bỏ climatology, đường chéo (cùng biến) vẫn chiếm ưu thế. Các liên hệ chéo lớn nhất ở mức trung bình cửa sổ gồm:

- nhiệt độ quá khứ → áp suất tương lai: `r = -0.266`;
- áp suất quá khứ → thành phần gió bắc-nam tương lai: `r = 0.233`;
- áp suất quá khứ → RH tương lai: `r = -0.220`;
- RH quá khứ → nhiệt độ tương lai: `r = -0.222`;
- thành phần gió bắc-nam quá khứ → nhiệt độ tương lai: `r = 0.324`;
- thành phần gió đông-tây quá khứ → RH tương lai: `r = -0.293`.

Các quan hệ này phù hợp để làm feature tương tác, nhưng chỉ là association theo thời gian, không phải bằng chứng nhân quả. Ma trận cũng không đối xứng vì một phía luôn là cửa sổ quá khứ và phía kia là cửa sổ tương lai.

## Quy luật nên mang sang mô hình

1. Dùng climatology theo trạm–mùa–giờ như baseline riêng; model động lực nên học anomaly thay vì phải học lại mùa vụ.
2. Ưu tiên mạnh 0–24h gần origin. Nhiệt độ/áp suất/RH vẫn đáng dùng đến khoảng 48h nhưng suy giảm rõ; gió lịch sử chủ yếu có ích ở horizon ngắn.
3. Không nên dùng một decay chung cho mọi biến. Gió cần decay nhanh hơn nhiệt độ, áp suất và RH.
4. Giữ `u/v` cho gió thay vì góc độ; cách này tránh discontinuity 0/360° và bảo toàn quan hệ vector.
5. Ở lead xa, chỉ lịch sử khí tượng không đủ để tái tạo thời tiết thực tế. Muốn dự báo 72h tốt cần forecast weather bên ngoài hoặc một weather decoder có uncertainty; persistence thuần sẽ tiến gần climatology.

## Giới hạn diễn giải

Cửa sổ chồng lấn làm số mẫu danh nghĩa lớn nhưng không tạo ra 11,641 mẫu độc lập. Vì mục tiêu ở đây là EDA/quy luật mô tả, báo cáo không gắn p-value ngây thơ cho từng cửa sổ. Các trạm cũng có phụ thuộc không gian. Nếu dùng kết quả để kiểm định mô hình hoặc viết confidence interval, nên block-bootstrap theo các block thời gian dài (ví dụ 1–4 tuần), không bootstrap từng origin.

## File kết quả

- `lead_correlation_curves.png`: decay theo lead, có IQR giữa các trạm.
- `offset_pair_anomaly_heatmaps.png`: toàn bộ ma trận 24 mốc quá khứ × 24 mốc tương lai.
- `cross_feature_correlations.png`: ma trận biến quá khứ × biến tương lai, raw và anomaly.
- `lead_correlations.csv`, `offset_pair_correlations.csv`, `cross_feature_window_correlations.csv`: bảng số đầy đủ.
- `wind_direction_persistence.csv`: persistence hướng gió theo metric vòng tròn.
- `summary.json`: tóm tắt machine-readable.
