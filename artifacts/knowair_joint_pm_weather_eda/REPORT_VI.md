# EDA cuối KnowAir: cấu trúc chung PM–khí tượng trong cửa sổ 72h → 72h

## Kết luận trung tâm

Phân tích toàn bộ KnowAir cho thấy một **sự chuyển giao thông tin theo horizon**:

> Ở vài giờ đầu, PM tương lai chủ yếu là phần tiếp diễn của trạng thái PM hiện tại. Từ 24–72h, memory PM suy yếu nhanh và sự tiến hóa của regime khí tượng—đặc biệt ở quy mô vùng—trở thành nguồn thông tin lớn hơn.

Đồng thời có một chuyển giao theo không gian:

> Gần origin, trạng thái riêng của trạm đích là nền tảng. Phần bổ sung từ các trạm khác có hướng theo gió, mạnh nhất ở +6h, và vẫn còn sau khi loại bớt PM vùng cùng khí tượng tương lai tại đích.

Các phát hiện mới quan trọng nhất là:

1. Adjusted `R²` median của PM history giảm từ `0.760` ở +3h xuống `0.010` ở +72h. Trái lại, realized future weather bổ sung `0.020 → 0.206`.
2. Với mode PM vùng thứ nhất, future weather mode bổ sung `R²=0.429` ở +72h, gấp khoảng 10 lần phần PM-mode history còn lại (`0.043`).
3. Onset và clearance không phải hai quá trình đối xứng. Onset đi cùng ventilation suy sụp dần; clearance là một cú chuyển nhanh với ventilation tăng mạnh và pressure đảo pha—dấu vết của một clearing regime/front.
4. Cạnh PM xuôi gió ở +6h còn partial `r=0.203` sau khi control PM tại đích, PM vùng và future weather tại đích; cạnh ngược gió chỉ `0.052`.
5. PM-history và weather-history tạo hai không gian analog chuyên biệt. Nối thẳng tất cả feature bằng một Euclidean metric làm **giảm** khả năng tìm analog PM.

Đây là EDA association, không phải chứng minh nhân quả. Negative control trong báo cáo được dùng chủ động để chỉ ra phần nào có thể là dấu vết của latent weather regime hoặc common forcing.

## 1. Thiết kế phân tích

- Toàn bộ KnowAir 2015–2018: `11,688` timestamp, cadence 3h, `184` trạm.
- Quét đủ `11,641` origin 72h quá khứ → 72h tương lai; cửa sổ được phép chồng lấn; không xét train/validation/test.
- PM `<=0` được coi là missing; PM dùng `log1p` trước khi khử climatology.
- Mọi biến động lực được chuyển thành anomaly sau khi trừ mean `station × month × dataset-clock-hour` trên toàn bộ dữ liệu.
- Weather core gồm temperature, pressure, RH 950 hPa, wind speed, `u/v`, PBL, ventilation=`wind speed × PBL`, và `log1p(precipitation)`.
- Thống kê nội trạm được tính độc lập cho từng trạm rồi tóm tắt bằng median/IQR trên 184 trạm.
- Analog dùng 2,000 cặp origin ngẫu nhiên mỗi trạm; hai origin cách nhau ít nhất 144h để hai khối 72h không đè lên nhau trong phép so sánh analog.
- Liên trạm dùng năm láng giềng gần nhất của mỗi target: 920 cạnh có hướng cho mỗi wind regime/lead, tổng `11,040` hàng thống kê.
- `Adjusted R²` và partial correlation là mô tả in-sample. Future weather thực đo là **oracle diagnostic**, không phải feature causal-available tại origin.

## 2. Độc lập tại mỗi trạm

### 2.1 Có một “information handoff” theo horizon

Mô hình tuyến tính theo từng trạm được phân rã thành:

1. PM history: PM cuối history, mean 24h gần nhất và trend giữa hai block 24h;
2. cộng toàn bộ past-weather state/mean/trend;
3. hoặc cộng realized future weather đúng tại lead, dùng như oracle chẩn đoán.

| Lead | `R²adj` PM history | Gain past weather | Gain realized future weather |
|---:|---:|---:|---:|
| +3h | 0.760 | 0.020 | 0.020 |
| +24h | 0.167 | 0.091 | 0.148 |
| +48h | 0.035 | 0.056 | 0.195 |
| +72h | 0.010 | 0.045 | 0.206 |

Ở +3h, PM history áp đảo mọi thứ khác. Đến +24h, past weather đã thêm hơn nửa lượng `R²` của PM history. Ở +48/+72h, weather trajectory chứa nhiều thông tin hơn hẳn phần memory PM còn lại.

Điểm cần đọc đúng: future-weather gain lớn không có nghĩa model ở origin đã biết tương lai. Nó đặt một **trần cơ chế**: sai số weather forecast sẽ là bottleneck ngày 2–3, trong khi ngày đầu bottleneck chủ yếu nằm ở mô tả đúng trạng thái PM ban đầu.

### 2.2 Weather nào liên hệ độc lập với PM tương lai?

Sau khi control PM hiện tại tại cùng trạm, các partial correlation lớn nhất là:

| Lead | Các quan hệ past weather → future PM nổi bật |
|---:|---|
| +3h | wind speed `-0.167`, ventilation `-0.156`, PBL `-0.136`, precipitation `-0.101`, wind-v `+0.112` |
| +24h | wind speed `-0.153`, ventilation `-0.135`, RH `-0.096`, PBL `-0.091`, precipitation `-0.081`, wind-v `+0.133` |
| +48h | pressure `+0.099`, RH `-0.097`, temperature `-0.082`, wind speed `-0.068` |
| +72h | pressure `+0.113`, temperature `-0.095`, RH `-0.082` |

Có hai phase khác nhau:

- 0–24h là phase **dispersion/removal**: gió, ventilation, PBL và mưa mang dấu âm rõ.
- 48–72h là phase **synoptic regime**: pressure/temperature/RH nổi lên, trong khi các biến khuếch tán nhanh đã mất phần lớn memory.

Sự đảo vai này giải thích vì sao một hệ số weather cố định cho mọi horizon thường không đủ.

### 2.3 Negative control: PM cũng “dự báo” được một ít weather

Nếu đảo chiều và đo past PM → future weather, vẫn control weather hiện tại, PM có partial correlation với future temperature `+0.101` ở +3h; một số tín hiệu nhỏ khác tồn tại ở wind/PBL. PM không thể gây ra temperature tương lai theo cơ chế hợp lý ở quy mô này.

Do đó, một phần quan hệ weather–PM hai chiều là do cả hai cùng encode latent regime, season residual, boundary-layer state hoặc common regional forcing. Ở +24h trở đi, negative-control effects đa số nhỏ (`|r|<0.08`), nhưng đủ để bác bỏ cách đọc nhân quả trực tiếp từ một correlation đơn lẻ.

### 2.4 Future weather chỉ “giải thích mất” một phần nhỏ persistence PM

Control đồng thời future ventilation, RH, precipitation và pressure:

| Lead | PM persistence thô | Sau control future weather | Mức giảm median |
|---:|---:|---:|---:|
| +3h | 0.872 | 0.854 | 0.018 |
| +24h | 0.407 | 0.380 | 0.027 |
| +48h | 0.187 | 0.172 | 0.013 |
| +72h | 0.096 | 0.093 | 0.006 |

PM persistence không đơn thuần là weather persistence được nhìn gián tiếp. Phần lớn memory PM còn lại sau control, phù hợp với đóng góp của tồn lưu aerosol, phát thải, chemistry chưa quan sát, địa hình và measurement continuity. Vì vậy model vẫn cần PM-state encoder riêng, không thể thay bằng weather encoder.

### 2.5 Bốn archetype cho thấy onset và clearance bất đối xứng

Event được xác định theo quintile anomaly riêng từng trạm. Tổng số station-origin:

| Event | Số origin | Median mỗi trạm |
|---|---:|---:|
| Clean persistence | 606,851 | 3,116 |
| Onset | 12,332 | 52 |
| Polluted persistence | 256,353 | 1,220 |
| Clearance | 28,853 | 138.5 |

Các trajectory là median của station-level mean, theo đơn vị standard deviation của từng trạm:

- **Onset:** PM từ `-1.181σ` ở −21h tăng lên `+0.061σ` ở +3h và `+1.074σ` ở +21h. Ventilation đã cao `+0.301σ` ở −21h nhưng rơi qua zero ở −3h và xuống `-0.160σ` ở +21h. Đây là một “closing gate”: khả năng pha loãng suy sụp trước khi PM đạt đỉnh.
- **Clearance:** PM từ `+1.127σ` ở −21h rơi xuống `-0.692σ` ở +3h và `-1.323σ` ở +9h. Ventilation đồng thời nhảy `-0.269σ → +0.922σ`; pressure đổi từ `-0.764σ` ở −21h thành `+0.498σ` ở +21h; RH cũng từ dương sang âm. Đây giống một lần quét sạch nhanh bởi chuyển regime/front hơn là mean reversion trơn.
- **Polluted persistence:** ventilation âm suốt vùng origin (`-0.299σ` ở −21h, `-0.273σ` ở +3h), pressure cũng âm, PM giữ trên `+1σ` quanh origin.
- **Clean persistence:** PM thấp đi cùng ventilation dương ổn định gần origin, không cần một cú weather shock lớn.

Quy luật ẩn là **trigger có hướng thời gian**: suy giảm ventilation là precursor của onset, còn tăng ventilation/pressure đột ngột là trigger clearance. Hai event head riêng hợp lý hơn giả định một dynamics đối xứng.

### 2.6 Analog cho thấy không nên nối PM và weather một cách ngây thơ

Metric analog là Spearman association giữa khoảng cách history và khoảng cách future; số lớn nghĩa “history giống nhau thì future cũng giống nhau hơn”. Median trên trạm:

| History/retrieval | Future tương ứng | Day 1 | Day 3 | Full 72h |
|---|---|---:|---:|---:|
| recent-24h PM-only | PM | 0.358 | 0.041 | 0.223 |
| recent-24h weather-only | Weather | 0.455 | 0.068 | 0.280 |
| recent-24h joint | Joint | 0.458 | 0.064 | 0.273 |
| full-72h PM-only | PM | 0.241 | 0.049 | 0.172 |
| full-72h weather-only | Weather | 0.292 | 0.060 | 0.190 |
| full-72h joint | Joint | 0.289 | 0.051 | 0.182 |

Hai điểm đặc biệt:

1. Recent 24h thường tốt hơn full 72h. History xa có thể hữu ích để nhận dạng regime, nhưng average distance trên toàn bộ 72h làm loãng trạng thái gần origin.
2. Với target PM day 1, PM-only đạt `0.358`, joint chỉ `0.196`, weather-only `0.116`. Weather là thông tin bổ sung trong regression, nhưng nếu trộn bằng một khoảng cách Euclidean đồng trọng số, nó lấn át geometry chuyên biệt của PM.

Hàm ý: dùng encoder/metric riêng cho PM và weather rồi fusion có học, thay vì concatenate rồi dùng một attention/distance duy nhất.

### 2.7 Quy luật thay đổi theo địa lý

Spearman theo latitude:

| Lead | PM-history `R²` | Past-weather gain | Future-weather gain |
|---:|---:|---:|---:|
| +3h | -0.374 | +0.710 | +0.710 |
| +24h | -0.597 | +0.056 | +0.554 |
| +48h | -0.624 | -0.605 | +0.484 |
| +72h | -0.585 | -0.720 | +0.452 |

Trạm phía bắc có PM persistence thấp hơn nhưng realized future weather hữu ích hơn. Với past weather, dấu theo latitude đổi từ dương ở horizon gần sang âm mạnh ở horizon xa. Đây không phải một quy luật vật lý đơn giản; nó cho thấy north/south đang có **khác biệt về availability của information**: phía bắc synoptic hơn nên cần biết weather trajectory tương lai, còn past weather khó extrapolate xa; ở phía nam, past regime có thể kéo dài hơn.

Model nên station-condition/horizon-condition weather gain và uncertainty, không dùng một fusion weight chung.

## 3. Quan hệ giữa các trạm

### 3.1 PM vùng cũng trải qua information handoff mạnh hơn PM cục bộ

PCA không gian cho PM cho thấy PC1 giải thích `25.6%` variance anomaly. Với PM mode thứ nhất:

| Lead | `R²adj` PM-mode history | Gain past-weather modes | Gain future-weather modes |
|---:|---:|---:|---:|
| +3h | 0.954 | 0.009 | 0.015 |
| +24h | 0.516 | 0.136 | 0.217 |
| +48h | 0.152 | 0.178 | 0.390 |
| +72h | 0.043 | 0.176 | 0.429 |

Đây là bằng chứng mạnh nhất cho scale separation: short horizon là continuation của aerosol field; far horizon là weather-regime transition. Kết quả tương tự xuất hiện ở năm PM components đầu, không chỉ PC1.

Past-weather modes vẫn thêm `0.176` ở +72h—lớn hơn nhiều so với PM-mode history—nhưng future weather cho trần cao hơn `0.429`. Khoảng cách giữa hai con số định lượng phần khó của bài toán: **dự báo đúng sự tiến hóa khí tượng vùng**, không chỉ đọc khí tượng quá khứ.

### 3.2 Dấu vết truyền PM theo hướng gió vẫn tồn tại sau ba tầng control

Với mỗi trạm đích, dùng năm láng giềng gần nhất (median khoảng cách `102.8 km`). Correlation nguồn PM → target future PM được tính trong ba mức:

1. control PM hiện tại tại target;
2. thêm control PM trung bình vùng;
3. thêm realized future ventilation/RH/rain/pressure tại target.

| Lead | Hướng | Control target PM | + regional PM | + target future weather |
|---:|---|---:|---:|---:|
| +3h | Xuôi gió | 0.231 | 0.207 | 0.199 |
|  | Ngang gió | 0.149 | 0.122 | 0.114 |
|  | Ngược gió | 0.081 | 0.056 | 0.051 |
| +6h | Xuôi gió | 0.241 | 0.206 | 0.203 |
|  | Ngang gió | 0.165 | 0.124 | 0.120 |
|  | Ngược gió | 0.091 | 0.056 | 0.052 |
| +12h | Xuôi gió | 0.189 | 0.142 | 0.142 |
|  | Ngang gió | 0.144 | 0.091 | 0.088 |
|  | Ngược gió | 0.085 | 0.042 | 0.040 |
| +24h | Xuôi gió | 0.104 | 0.056 | 0.051 |
|  | Ngang gió | 0.086 | 0.039 | 0.032 |
|  | Ngược gió | 0.057 | 0.010 | 0.007 |

Regional control làm giảm tín hiệu, chứng tỏ common regional plume/forcing giải thích một phần. Tuy nhiên ở +6h, cạnh xuôi gió còn `0.203`, gần bốn lần ngược gió `0.052`, và gần như không giảm thêm khi control future weather (`0.206 → 0.203`). Đây là pattern phù hợp với transport/advection hoặc nguồn phát thải không gian có hướng, nhưng vẫn chưa đủ để gọi là causal transport vì inventory phát thải và back-trajectory không có trong EDA.

### 3.3 Láng giềng được align theo gió có value-add thực

So sánh gain adjusted `R²` của aggregated PM-neighbor với baseline target PM:

| Lead | Wind-aligned gain median [IQR] | Unweighted-neighbor gain |
|---:|---:|---:|
| +3h | 0.0150 [0.0087, 0.0216] | 0.0103 |
| +6h | 0.0272 [0.0173, 0.0389] | 0.0199 |
| +12h | 0.0213 [0.0131, 0.0324] | 0.0161 |
| +24h | 0.0050 [0.0008, 0.0105] | 0.0029 |

Gain peak ở +6h và wind alignment luôn tốt hơn trung bình láng giềng không trọng số. Vì vậy adjacency chỉ theo khoảng cách bỏ mất signal có hệ thống. Graph nên là directed, time-varying và wind-gated; hiệu ứng cạnh nên decay mạnh sau 12–24h.

### 3.4 Hai scale phải được model đồng thời

Kết quả PCA và neighbor graph không mâu thuẫn:

- Regional modes mô tả nền weather/PM có coherence rộng và chi phối horizon xa.
- Directed neighbors mô tả phần local transport/residual, hữu ích nhất ở 3–12h.

Một graph dày cố gắng học cả hai bằng cùng một message-passing kernel dễ trộn common mode với transport. Cấu trúc tự nhiên hơn là `regional latent state + local directed residual graph`.

## 4. Các quy luật ẩn cô đọng

1. **Temporal information handoff:** PM state → past weather context → future weather evolution khi horizon tăng.
2. **Spatial information handoff:** target state → wind-aligned neighbor residual ở 3–12h; regional modes → far-horizon evolution.
3. **Precursor–trigger asymmetry:** ventilation collapse đi trước onset; ventilation/pressure jump kích hoạt clearance nhanh.
4. **Persistence không phải mediation đơn giản:** control future weather chỉ làm giảm PM persistence rất ít; aerosol/source state cần latent riêng.
5. **Representation specialization:** PM-nearness và weather-nearness là hai geometry khác nhau; naive concatenation làm hỏng analog retrieval.
6. **Recent-state dominance:** recent 24h giàu thông tin hơn full 72h nếu dùng metric/pooling đồng đều; phần 24–72h nên dùng như regime context có trọng số thấp hoặc multi-scale branch.
7. **Geographic skill shift:** trạm phía bắc ít PM memory hơn và phụ thuộc mạnh hơn vào future-weather accuracy.
8. **Causal humility:** negative control PM→weather cho thấy latent regime confounding; hướng gió và nhiều tầng control làm transport hypothesis mạnh hơn, nhưng không biến nó thành bằng chứng nhân quả.

## 5. Hàm ý trực tiếp cho model

1. Dùng **horizon-gated mixture**: PM-persistence expert cho 0–12h, weather-transition expert tăng trọng số từ 24–72h.
2. Tách PM encoder và weather encoder; fusion bằng learned gate/cross-attention thay vì concatenate đồng trọng số.
3. Tách recent-24h branch khỏi 24–72h regime branch; không average/pool đều toàn history.
4. Dùng latent regional PM/weather module cho horizon xa và local correction cho từng trạm.
5. Dùng dynamic directed graph với edge weight theo distance, wind alignment và lead; prior peak khoảng +6h cho láng giềng cỡ 100 km.
6. Thêm onset/clearance head hoặc change-point gate dựa trên đạo hàm ventilation, pressure và RH; không giả định dynamics tăng/giảm đối xứng.
7. Condition fusion/uncertainty theo station hoặc latitude. North stations cần nhạy hơn với weather forecast uncertainty.
8. Nếu future weather do weather decoder cung cấp, propagate uncertainty sang PM head; oracle gain ở +72h cho thấy lỗi weather có thể thống trị lỗi PM.
9. Ablation cần so sánh: PM-only, weather-only, joint-separate-encoder; static graph, distance graph, wind-directed graph; regional-only, local-only và hai tầng kết hợp.

## 6. Giới hạn

- Các cửa sổ chồng lấn nên không được xem là 11,641 quan sát IID; không dùng p-value ngây thơ theo window.
- Climatology và PCA fit trên toàn bộ dữ liệu vì mục tiêu là EDA toàn bộ. Khi đánh giá forecast, chúng phải fit chỉ trên train.
- Adjusted `R²` là in-sample descriptive, không phải out-of-sample forecast skill.
- Partial correlation loại confounder tuyến tính đã quan sát, không loại được emission, terrain, chemistry hay latent nonlinear regimes.
- Future weather là oracle diagnostic; ở deployment cần numerical weather forecast hoặc weather decoder.
- Event thresholds theo quintile tạo archetype tương đối giữa trạm, không phải ngưỡng sức khỏe PM tuyệt đối.
- Analog metric đo distance-rank association, không phải forecast error; kết luận về representation áp dụng trực tiếp nhất cho retrieval/distance-based models.
- PM missing được điền zero sau chuẩn hóa chỉ trong spatial PCA; vì zero khi đó là anomaly trung bình và missing chỉ khoảng 0.375%, ảnh hưởng dự kiến nhỏ nhưng vẫn nên kiểm tra bằng PCA có mask trong nghiên cứu tiếp theo.

## 7. Artifacts

- `within_station_joint_information.png`: cross-lag, information handoff và mediation tại từng trạm.
- `joint_event_trajectories.png`: bốn archetype clean/onset/polluted/clearance.
- `cross_station_joint_transport.png`: transport theo hướng gió sau các tầng control và dynamic-neighbor gain.
- `joint_analogs_and_regional_modes.png`: analog specialization và coupling PM–weather modes.
- `within_station_cross_lag_matrix.csv`: đủ 400 cặp feature/lead.
- `station_pm_weather_value_add.csv`: decomposition cho từng trạm và lead.
- `joint_event_composites.csv`: trajectory event theo 25 offset từ −69h đến +72h.
- `cross_station_joint_transport_edges.csv`: đủ 11,040 edge/regime/lead records.
- `joint_analog_predictability.csv`: 9,936 kết quả analog theo trạm.
- `regional_pm_weather_mode_coupling.csv`: năm PM modes tại bốn lead.
- `summary.json`: tóm tắt machine-readable.

