# EDA sâu KnowAir: động lực PM2.5, khí tượng và vận chuyển không gian

## Executive summary

Phân tích sâu trên toàn bộ 2015–2018 cho thấy PM2.5 trong KnowAir được chi phối bởi năm cấu phần có thang thời gian khác nhau:

1. **Persistence PM cục bộ rất mạnh nhưng ngắn**: PM gần origin có `r=0.872` với +3h, còn `0.407` ở +24h, `0.187` ở +48h và `0.096` ở +72h sau khi bỏ climatology.
2. **Mean reversion rõ và bất đối xứng**: trạng thái PM rất cao thường giảm nhanh hơn trạng thái sạch tăng lên. Xu hướng tăng/giảm chỉ có giá trị gia tăng nhỏ ở ngày 1 và gần như mất hết từ ngày 2.
3. **Thông gió là cơ chế khí tượng mạnh nhất**: gió, chiều cao lớp biên (PBL), và tích `wind × PBL` đều liên hệ âm, gần đơn điệu với PM. Mưa mạnh có hiệu ứng làm sạch rõ.
4. **PM lan truyền theo cạnh không gian có hướng**: PM láng giềng xuôi gió bổ sung thông tin vượt PM hiện tại tại trạm đích; tín hiệu mạnh hơn 2–3 lần cạnh ngược gió và peak dịch theo khoảng cách.
5. **Climatology tuyệt đối rất mạnh nhưng khác với động lực dự báo**: mùa đông ô nhiễm cao hơn mùa hè nhiều lần. Mọi đánh giá forecast nên tách climatology trạm–tháng–giờ khỏi anomaly.

Đây là association EDA, không phải bằng chứng nhân quả. Tuy nhiên, sự lặp lại theo năm, tính đơn điệu của ventilation regime, và dịch chuyển độ trễ theo khoảng cách làm các quy luật cơ chế đáng tin hơn tương quan thô đơn lẻ.

## 1. Thiết kế phân tích

- Toàn bộ KnowAir: 11,688 timestamp cách nhau 3h, 184 trạm, từ 2015-01-01 đến 2018-12-31.
- 11,641 origin gồm 72h quá khứ và 72h tương lai, cho phép chồng lấn; không dùng split train/validation/test.
- `PM <= 0` được coi là missing, chiếm 0.375% dữ liệu. Đây phù hợp với convention target-valid hiện có trong repo hơn là diễn giải PM bằng 0.
- Động lực PM được phân tích trên `log1p(PM)` rồi trừ climatology riêng theo `station × month × dataset-clock-hour`.
- Tương quan được tính riêng từng trạm rồi lấy median/IQR trên 184 trạm.
- Với thời tiết tương lai thực đo, báo cáo ghi rõ đây là diagnostic cơ chế/oracle; model causal tại origin không tự có các biến này.
- Hướng gió dùng `u/v`, không dùng Pearson trực tiếp trên góc 0–360°.

## 2. PM có ba thang “memory”

### 2.1 Giá trị gần nhất chi phối 0–24h

Median station correlation giữa PM cuối quá khứ và PM tương lai trên log-anomaly:

| Lead | 3h | 12h | 24h | 48h | 72h |
|---|---:|---:|---:|---:|---:|
| `last PM` | 0.872 | 0.578 | 0.407 | 0.187 | 0.096 |
| trung bình PM 24h | 0.691 | 0.496 | 0.328 | 0.160 | 0.085 |
| trung bình PM 72h | 0.469 | 0.345 | 0.238 | 0.121 | 0.078 |

`Last PM` xuống dưới `r=0.5` từ 18h; trung bình 24h từ 12h; trung bình 72h đã dưới 0.5 ngay ở +3h. Điều này không có nghĩa lịch sử xa vô ích, nhưng nó nên đóng vai trò xác định regime/baseline thay vì được model xem ngang hàng với vài bước gần origin.

### 2.2 Xu hướng chỉ thêm thông tin ngắn hạn

Xu hướng được định nghĩa bằng mean log-anomaly 24h gần nhất trừ 24h trước đó. Nếu tương quan trực tiếp xu hướng với “future change”, kết quả âm khá mạnh, nhưng metric này dùng chung số hạng PM gần nhất ở hai phía và dễ tạo coupling toán học. Thước đo sạch hơn là partial correlation của xu hướng với PM tương lai sau khi control mức PM 24h gần nhất:

| Target | r(xu hướng, future level) | partial r sau control recent PM |
|---|---:|---:|
| Ngày 1 | 0.337 | 0.100 |
| Ngày 2 | 0.136 | 0.025 |
| Ngày 3 | 0.058 | -0.002 |

Kết luận: slope/trend có ích như feature phụ cho ngày đầu, nhưng gần như không mang value-add độc lập ở ngày 2–3.

### 2.3 Mùa thay đổi độ bền, không thay quy luật

Ở +24h, persistence PM là 0.418 (DJF), 0.335 (MAM), 0.411 (JJA), 0.431 (SON). Ở +72h, JJA còn 0.152 trong khi MAM chỉ 0.051. Mức PM mùa hè thấp hơn nhiều nhưng anomaly lại giữ memory xa hơn đôi chút. Quy luật giảm theo lead vẫn tồn tại ở cả bốn mùa.

Theo từng năm, `last PM → future PM` ở +3h nằm trong 0.863–0.871. Ở +24h là 0.332–0.401. Vì vậy persistence ngắn hạn không do riêng một năm bất thường.

## 3. Mean reversion và chuyển trạng thái ô nhiễm

Khi chia trạng thái PM gần nhất thành năm quintile anomaly, median PM change đến +72h khoảng:

- quintile thấp nhất: tăng khoảng 16 µg/m³;
- quintile thứ hai: tăng khoảng 7 µg/m³;
- quintile giữa: gần 0;
- quintile thứ tư: giảm khoảng 8 µg/m³;
- quintile cao nhất: giảm khoảng 29 µg/m³.

Đây là regression-to-the-mean có điều kiện, không nên gọi là hiệu ứng nhân quả. Tuy nhiên nó cho thấy forecast deterministic giữ nguyên PM sẽ bị bias: dự báo quá thấp sau trạng thái cực sạch và quá cao sau spike lớn.

### Ma trận chuyển trạng thái 24h

Xác suất còn ở đúng category vào ngày tương lai thứ nhất / thứ ba:

| Category mean PM 24h | Day 1 | Day 3 |
|---|---:|---:|
| ≤35 | 71.7% | 58.1% |
| 35–75 | 60.8% | 49.1% |
| 75–150 | 50.4% | 33.6% |
| >150 | 47.2% | 22.2% |

Trạng thái sạch “dính” hơn trạng thái cực xấu. Nếu mean PM ngày gần nhất >150, sang day 1 có 36.1% rơi về 75–150 và chỉ 47.2% còn >150; đến day 3 chỉ 22.2% còn >150.

### Độ dài episode

| Ngưỡng | Median | P90 | P95 | P(duration >24h) | P(duration >48h) |
|---|---:|---:|---:|---:|---:|
| PM >35 | 9h | 66h | 102h | 23.7% | 13.1% |
| PM >75 | 6h | 33h | 54h | 13.8% | 5.8% |
| PM >150 | 6h | 24h | 39h | 9.2% | 2.9% |

Episode cực trị phần lớn ngắn, nhưng đuôi phân phối dài. Điều này ủng hộ loss/head riêng cho xác suất duy trì episode thay vì chỉ tối ưu MAE trung bình.

## 4. Khí tượng nào thật sự liên hệ với PM tương lai?

### 4.1 So sánh PM history, past weather và realized future weather

Median station `r` với future PM anomaly:

| Predictor | +3h | +24h | +48h | +72h |
|---|---:|---:|---:|---:|
| PM gần nhất | 0.872 | 0.407 | 0.187 | 0.096 |
| Wind speed gần nhất | -0.280 | -0.233 | -0.107 | -0.033 |
| Ventilation gần nhất | -0.299 | -0.227 | -0.091 | -0.022 |
| PBL gần nhất | -0.253 | -0.172 | -0.057 | -0.007 |
| Mưa gần nhất | -0.130 | -0.094 | -0.042 | -0.022 |

PM history vẫn là predictor riêng lẻ mạnh nhất trong ngày đầu. Past wind/ventilation/PBL bổ sung tín hiệu làm sạch và giảm chậm hơn một số biến khí tượng khác.

Quan hệ contemporaneous giữa realized weather và PM anomaly gần như không đổi theo lead vì cùng một quan hệ được dịch dọc chuỗi: ventilation `r≈-0.257`, wind speed `-0.229`, PBL `-0.211`, precipitation `-0.107`, pressure `-0.179`, temperature `+0.213`. Quan hệ ventilation–PM lặp lại rất ổn định từng năm (`-0.248` đến `-0.287`).

### 4.2 Quy luật “ventilation gate” gần đơn điệu

Trong quintile PM ban đầu cao nhất, khi realized future ventilation đi từ quintile thấp nhất lên cao nhất:

- median PM change 72h chuyển từ `+0.4` thành `-24.0 µg/m³`;
- xác suất future mean PM >75 giảm từ `44%` xuống `17%`.

PBL cho xác suất tương ứng `46% → 16%`; wind speed `44% → 21%`. Ở mọi mức PM ban đầu, tăng wind/PBL/ventilation đều dịch PM change theo hướng giảm. Đây là một trong các quy luật phi tuyến rõ nhất bộ dữ liệu: **PM state quyết định “bao nhiêu có thể tích tụ”, ventilation quyết định episode được duy trì hay bị xả**.

Hàm ý là interaction `PM × ventilation` quan trọng hơn chỉ cộng tuyến tính từng biến. Một gating/source-sink module phù hợp hơn một hệ số weather cố định.

### 4.3 Mưa có threshold vật lý

Khi dùng tổng mưa thực trong 72h thay vì anomaly quintile, ở quintile PM ban đầu cao nhất:

| Tổng mưa tương lai | ≤0.1 mm | 0.1–1 mm | 1–5 mm | >5 mm |
|---|---:|---:|---:|---:|
| Median PM change | -7.5 | -6.3 | -11.7 | -16.3 |
| P(future mean PM >75) | 47% | 37% | 27% | 10% |

Ở quintile PM thứ tư, xác suất >75 giảm từ 32% khi gần như khô xuống 4% khi mưa >5 mm. Mưa nhỏ không phải lúc nào cũng đủ làm sạch; hiệu ứng mạnh và ổn định hơn ở nhóm >5 mm.

### 4.4 RH là phi tuyến và bị confounding

Linear contemporaneous relation RH–PM anomaly khá yếu (`r≈-0.053`). Nhưng trong trạng thái PM cao nhất, xác suất future mean >75 tăng từ khoảng 31% ở bốn RH quintile đầu lên 41% ở quintile RH cao nhất. Điều này có thể phản ánh hygroscopic growth, stagnant humid regimes, hoặc đồng biến với yếu tố khác; không nên ép một hệ số RH tuyến tính chung.

## 5. Dấu vết vận chuyển PM theo gió

Với mỗi trạm đích, lấy năm láng giềng gần nhất (920 cạnh có hướng). Metric là partial correlation:

`PM_source(t) ↔ PM_target(t+lag) | PM_target(t)`

sau đó chia theo wind alignment tại nguồn. Median khoảng cách cạnh là 103 km.

| Lead | Xuôi gió | Ngang gió | Ngược gió |
|---|---:|---:|---:|
| +3h | 0.231 | 0.149 | 0.081 |
| +6h | 0.241 | 0.165 | 0.091 |
| +12h | 0.189 | 0.144 | 0.085 |
| +24h | 0.104 | 0.086 | 0.057 |

Tín hiệu nguồn xuôi gió mạnh hơn rõ so với ngược gió dù đã control PM hiện tại tại đích. Quan trọng hơn, peak dịch theo khoảng cách:

- `<50 km`: peak ở +3h (`r=0.260`);
- `50–100 km`: +3/+6h (`0.259`);
- `100–200 km`: peak ở +6h (`0.238`);
- `≥200 km`: peak ở +6h (`0.168`), với +12h vẫn gần tương đương (`0.151`).

100 km trong 6h tương đương vận tốc hiệu dụng khoảng 4.6 m/s, cùng bậc với gió 100m trong dữ liệu. Đây là inference phù hợp vật lý, chưa phải ước lượng vận tốc plume. Common regional forcing, độ phân giải khí tượng và topology chỉ dùng năm láng giềng vẫn có thể gây confounding.

## 6. Climatology PM tuyệt đối

Median PM theo tháng (trung bình các cell giờ): tháng 1 là 70 µg/m³, tháng 12 là 66, trong khi tháng 7–9 chỉ 28–30. Xác suất PM >75 là 45.8% ở tháng 1, 43.1% tháng 12, nhưng 3.4% tháng 8.

Theo clock của dataset, median thấp nhất quanh hour 9 (36) và cao nhất quanh hour 15 (48). Metadata hiện tại không đủ để khẳng định clock này là UTC hay local, nên không gắn diễn giải phát thải theo giờ địa phương.

Climatology mạnh giải thích vì sao tương quan raw ở horizon dài cao giả tạo. Model nên có baseline climatology riêng rồi dự báo anomaly/dynamic correction.

## 7. Kiến trúc/quy luật nên thử trong model

1. **Hai nhánh baseline + anomaly**: climatology `station × month × hour` và dynamic residual `log1p(PM)`.
2. **Recency kernel theo horizon**: last 3–24h có trọng số lớn; history 24–72h chủ yếu encode regime. Không dùng pooling đều 72h.
3. **Variable-specific memory**: PM, nhiệt độ, áp suất, RH và gió có decay khác nhau; không dùng chung temporal filter.
4. **Ventilation-gated source/sink**: tương tác PM state với `wind × PBL`, rain threshold và RH nonlinear.
5. **Dynamic directed graph**: cạnh source→target được gate bằng alignment gió; kernel thời gian phụ thuộc distance, khoảng 3h cho cạnh gần và 6–12h cho cạnh xa.
6. **Trend feature chỉ cho short head**: slope có value-add ở day 1, không nên buộc ảnh hưởng day 2–3.
7. **Probabilistic episode head**: dự báo xác suất giữ/chuyển category và hazard kết thúc episode, đặc biệt cho >75 và >150.
8. **Future-weather uncertainty**: realized future ventilation rất hữu ích nhưng không causal-available. Khi dùng forecast weather, nên propagate uncertainty vào PM head thay vì coi weather decoder là ground truth.
9. **Missing mask rõ ràng**: không biến PM=0 thành quan sát sạch.

## 8. Giới hạn và cách đọc đúng

- 11,641 cửa sổ chồng lấn không phải 11,641 mẫu độc lập; không dùng p-value IID theo origin.
- 184 trạm có phụ thuộc không gian. IQR giữa trạm mô tả heterogeneity, không phải confidence interval độc lập.
- Climatology/anomaly được fit trên toàn bộ dữ liệu vì đây là EDA toàn bộ theo yêu cầu; không được làm vậy khi đánh giá forecast ngoài mẫu.
- Regime dùng realized future weather là diagnostic cơ chế, không phải feature hợp lệ tại thời điểm forecast nếu chưa có weather forecast.
- Mean reversion có một phần do chọn trạng thái cực trị và measurement noise. Không diễn giải thành lực vật lý tự động kéo PM về trung bình.
- Direction-conditioned partial correlation vẫn có thể giữ common-driver confounding; cần ablation forecast hoặc quasi-experiment để khẳng định causal transport.
- Các ngưỡng 35/75/150 µg/m³ ở đây là bin mô tả, không tuyên bố mapping sang chuẩn AQI cụ thể.

## 9. Artifacts

- `pm_persistence_and_mean_reversion.png`: memory và regression-to-mean.
- `weather_to_future_pm_by_lead.png`: past/future weather association theo lead.
- `joint_pm_weather_regime_heatmaps.png`: interaction PM state × weather regime.
- `seasonal_and_spatial_diagnostics.png`: độ bền theo mùa và wind-conditioned neighbours.
- `downwind_transport_by_distance.png`: dịch peak transport theo khoảng cách.
- `pm_month_hour_climatology.png`: climatology tuyệt đối.
- Các CSV cùng thư mục chứa số liệu đầy đủ; `summary.json` là bản machine-readable.
