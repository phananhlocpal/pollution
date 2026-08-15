# EDA cuối: cấu trúc ẩn của khí tượng 72h quá khứ → 72h tương lai

## Kết luận trung tâm

Khí tượng KnowAir không tuân theo một quy luật persistence đồng nhất. Cấu trúc tốt nhất để mô tả dữ liệu là:

> **Future weather = regional slow modes + station-local fast residuals**, trong đó thông tin hữu ích tập trung rất mạnh ở 24h gần origin, và tốc độ mất memory phụ thuộc biến, mùa, vĩ độ và hướng không gian.

Ba quy luật ẩn nổi bật nhất:

1. **Regionality–recency:** 24h gần nhất quan trọng hơn toàn bộ 72h; đồng thời các mode không gian cấp vùng bền hơn residual cục bộ rất nhiều.
2. **Geographic memory gradient:** trạm càng về phía bắc, persistence 24h của pressure/temperature/RH/wind càng thấp. Spearman theo vĩ độ là `-0.91` cho pressure và `-0.72` cho temperature.
3. **Directed anisotropy:** sau khi control thời tiết hiện tại tại trạm đích, trạm nằm phía tây/tây-bắc của đích bổ sung tín hiệu mạnh cho pressure tương lai theo hướng nguồn→đích đông/đông-nam. Pattern lặp lại trong cả bốn năm.

Các kết quả này gợi ý một model tốt không nên dùng một temporal encoder và một adjacency tĩnh duy nhất cho tất cả trạm.

## 1. Thiết kế EDA

- Toàn bộ KnowAir 2015–2018: 11,688 timestamp, cadence 3h, 184 trạm.
- Quét đủ 11,641 cửa sổ 72h quá khứ → 72h tương lai; cửa sổ được phép chồng lấn; không dùng split train/validation/test.
- Phân tích động lực trên anomaly sau khi trừ mean `station × month × dataset-clock-hour`.
- Biến nội trạm: temperature, pressure, RH 950 hPa, wind speed, `u/v`, PBL, ventilation, dewpoint deficit và `log1p(precipitation)`.
- Tất cả Pearson persistence được tính độc lập ở từng trạm rồi mới lấy median/IQR trên 184 trạm.
- Analog analysis dùng 2,500 cặp origin cho mỗi trạm; hai cửa sổ cách nhau ít nhất 144h để không dùng các đoạn dữ liệu chồng lên nhau.
- Liên trạm dùng 235,704 directed edges (`184 × 183 × 7 biến`). Partial correlation nguồn→đích control giá trị hiện tại tại trạm đích.

## 2. Trường hợp độc lập mỗi trạm

### 2.1 Mỗi biến có một đồng hồ memory riêng

Median station autocorrelation trên anomaly:

| Biến | +3h | +24h | +48h | +72h | Lead đầu tiên dưới 0.5 (median trạm) |
|---|---:|---:|---:|---:|---:|
| Temperature | 0.929 | 0.648 | 0.381 | 0.211 | 33h |
| Pressure | 0.985 | 0.622 | 0.261 | 0.089 | 33h |
| RH 950 hPa | 0.931 | 0.511 | 0.253 | 0.148 | 27h |
| Dewpoint deficit | 0.879 | 0.526 | 0.275 | 0.180 | 12h* |
| Wind speed | 0.817 | 0.205 | 0.034 | -0.012 | 9h |
| Wind u | 0.887 | 0.288 | 0.041 | -0.028 | 15h |
| Wind v | 0.903 | 0.208 | 0.002 | -0.010 | 15h |
| Ventilation | 0.736 | 0.244 | 0.058 | 0.004 | 6h |
| PBL | 0.591 | 0.358 | 0.142 | 0.075 | 6h |
| Log precipitation | 0.650 | 0.093 | 0.013 | 0.002 | 6h |

`*` Dewpoint deficit không giảm đơn điệu: correlation xuống dưới 0.5 rồi hồi lại ở gần chu kỳ 24h. Temperature cũng có một vai/rebound nhẹ quanh 15–24h. Sau khi đã bỏ mean theo giờ, daily echo còn lại có thể đến từ persistence của biên độ/pha ngày đêm, không phải climatology mean đơn giản.

Quy luật: pressure/temperature/RH là slow state; wind/PBL/rain là fast state. Không nên dùng chung decay kernel cho các nhóm này.

### 2.2 Trong 72h quá khứ, “độ tuổi” dữ liệu quan trọng hơn độ dài context

Tương quan của từng mốc quá khứ với mean của từng future day:

| Feature / past offset | Future day 1 | Day 2 | Day 3 |
|---|---:|---:|---:|
| Temperature, 0h | 0.803 | 0.505 | 0.291 |
| Temperature, -24h | 0.503 | 0.289 | 0.158 |
| Temperature, -48h | 0.288 | 0.156 | 0.079 |
| Temperature, -69h | 0.168 | 0.085 | 0.031 |
| Pressure, 0h | 0.857 | 0.412 | 0.161 |
| Pressure, -24h | 0.410 | 0.156 | 0.031 |
| Wind speed, 0h | 0.532 | 0.112 | -0.001 |
| Wind speed, -24h | 0.112 | -0.001 | -0.017 |

Với wind speed, dữ liệu cũ hơn 24h gần như không còn liên hệ với day 2–3. Với temperature, history xa vẫn còn tín hiệu nhỏ nhưng giảm gần theo tuổi. Cửa sổ 72h hữu ích như multi-scale context, không nên được average/pool đều.

### 2.3 Trend có value-add khác nhau theo biến

Trend là mean 24h gần nhất trừ mean 24h trước đó. Partial correlation với future-day level sau khi control recent state:

| Biến | Day 1 | Day 2 | Day 3 |
|---|---:|---:|---:|
| Pressure | 0.320 | 0.142 | 0.098 |
| Temperature | 0.236 | 0.126 | 0.076 |
| RH | 0.150 | 0.014 | -0.014 |
| Wind u | 0.159 | 0.066 | 0.020 |
| Wind v | 0.174 | 0.046 | 0.030 |
| Wind speed | 0.096 | 0.045 | 0.013 |
| PBL | 0.050 | -0.002 | -0.015 |
| Rain | 0.047 | 0.005 | -0.014 |

Pressure và temperature có inertia/trend có thể extrapolate qua nhiều ngày. RH trend chỉ hữu ích ngày đầu; PBL/rain trend gần như không có value-add ngoài current regime.

### 2.4 Regime transition có hình chữ U

Khi chia riêng mỗi trạm thành quintile, probability còn đúng quintile ở future day 1 thường cao nhất tại hai cực và thấp ở giữa:

- Temperature: Q1 `0.619`, Q3 `0.337`, Q5 `0.620`.
- Pressure: Q1 `0.549`, Q3 `0.326`, Q5 `0.598`.
- RH: Q1 `0.567`, Q3 `0.301`, Q5 `0.534`.
- Wind speed: Q1 `0.361`, Q3 `0.225`, Q5 `0.396`.

Extreme regimes “dính” hơn middle regimes: trạng thái rất nóng/lạnh, áp cao/thấp hoặc RH cao/thấp thường đại diện một hệ thời tiết ổn định; vùng giữa dễ chuyển quintile do gần nhiều boundary hơn. Đến day 3, transition gần random `0.2` hơn, nhưng temperature extremes vẫn còn `0.347/0.319`.

Rain có bất đối xứng khác: dry/low-rain state bền (`0.678` ở Q1 day 1), nhưng high-rain state chỉ `0.397`, phù hợp với rain event ngắt quãng.

### 2.5 Analog weather không quyết định tương lai xa

Spearman correlation giữa khoảng cách hai history trajectory và khoảng cách hai future trajectory:

| History dùng để tìm analog | Future day 1 | Future 72h | Future day 3 |
|---|---:|---:|---:|
| 24h gần nhất | 0.445 | 0.276 | 0.066 |
| Toàn bộ 72h | 0.284 | 0.184 | 0.056 |

Hai lịch sử rất giống nhau có xu hướng sinh ra day 1 giống nhau, nhưng gần như không ràng buộc day 3. Đáng chú ý, dùng toàn bộ 72h làm analog còn kém dùng 24h gần nhất: đoạn cũ thêm noise/dilution nhiều hơn signal.

### 2.6 Weather memory có gradient địa lý rất lớn

Spearman giữa vĩ độ trạm và persistence `r(+24h)`:

| Biến | ρ với vĩ độ | Range r(+24h) giữa 184 trạm |
|---|---:|---:|
| Pressure | -0.914 | 0.489–0.733 |
| Temperature | -0.716 | 0.577–0.709 |
| RH | -0.499 | 0.213–0.673 |
| Wind speed | -0.486 | 0.037–0.352 |

Trạm phía nam giữ memory lâu hơn; trạm phía bắc biến động/synoptic hơn. Đây là lý do mạnh để dùng station-conditioned decay, latitude-aware embedding hoặc mixture-of-dynamics thay vì một recurrent transition dùng chung tuyệt đối.

### 2.7 Season tạo một “predictability regime” riêng

Summer JJA có persistence xa cao bất ngờ:

- Pressure +72h: JJA `0.265`, DJF `0.158`, MAM `0.013`, SON `0.024`.
- PBL +72h: JJA `0.172`, DJF `0.025`, MAM `0.011`.
- Wind speed +24h: JJA `0.336`, DJF `0.171`, MAM `0.140`.

Spring MAM nhìn chung là regime khó dự báo xa nhất; summer có field ổn định hơn dù đối với PM đây không nhất thiết là mùa ô nhiễm cao. Season embedding nên điều khiển cả memory/uncertainty, không chỉ mean climatology.

## 3. Trường hợp quan hệ giữa các trạm

### 3.1 Weather field có correlation length rất khác theo biến

Median synchronous correlation theo khoảng cách:

| Biến | <50 km | 100–200 km | 400–800 km | 800–1600 km | ≥1600 km |
|---|---:|---:|---:|---:|---:|
| Pressure | 0.999 | 0.983 | 0.829 | 0.616 | 0.375 |
| Temperature | 0.988 | 0.919 | 0.657 | 0.484 | 0.361 |
| RH | 0.975 | 0.827 | 0.423 | 0.193 | 0.074 |
| Wind speed | 0.950 | 0.669 | 0.212 | 0.063 | 0.029 |
| PBL | 0.950 | 0.729 | 0.291 | 0.108 | 0.057 |

Pressure gần như là field cấp lục địa; temperature là field cấp vùng lớn; RH/PBL/wind localize nhanh hơn. Một graph radius chung là không hợp lý: pressure edges có thể rất xa, wind/PBL nên ưu tiên locality.

### 3.2 Field khí tượng có cấu trúc low-rank mạnh

Cumulative spatial variance giải thích bởi 1 / 3 / 10 PC:

| Biến | PC1 | Top 3 | Top 10 |
|---|---:|---:|---:|
| Pressure | 0.749 | 0.940 | 0.991 |
| Temperature | 0.617 | 0.789 | 0.918 |
| RH | 0.406 | 0.625 | 0.809 |
| Wind v | 0.361 | 0.579 | 0.793 |
| Wind u | 0.370 | 0.562 | 0.747 |
| PBL | 0.300 | 0.495 | 0.718 |
| Wind speed | 0.243 | 0.432 | 0.657 |

Pressure và temperature đặc biệt phù hợp với latent regional state nhỏ. Wind/PBL cần nhiều local modes hơn.

### 3.3 Regional mode giữ memory, local residual quên nhanh

Sau spatial PCA, so sánh total với top-3 regional reconstruction và residual sau khi bỏ top 3:

| Feature, +24h | Total | Regional top 3 | Local residual |
|---|---:|---:|---:|
| Temperature | 0.648 | 0.726 | 0.298 |
| Pressure | 0.622 | 0.646 | 0.189 |
| RH | 0.511 | 0.656 | 0.208 |
| Wind speed | 0.205 | 0.328 | 0.091 |
| Wind u | 0.288 | 0.373 | 0.104 |
| PBL | 0.358 | 0.482 | 0.204 |

Ở +72h, temperature regional còn `0.249` nhưng local chỉ `0.074`; RH `0.192` so với `0.047`. Phần predictability xa chủ yếu là regional mode. Local component có giá trị ở horizon gần rồi decay nhanh.

Đây là bằng chứng trực tiếp cho decomposition kiểu `global/regional latent dynamics + local correction`.

### 3.4 Directed lead–lag network bất đối xứng mạnh

Metric liên trạm là:

`corr(source(t), target(t+lead) | target(t))`

nên đo source station bổ sung gì ngoài persistence tại target. Với pressure, khoảng cách 200–400 km và lead +6h:

| Hướng source → target | Partial r |
|---|---:|
| SE | 0.466 |
| S | 0.397 |
| E | 0.385 |
| SW | 0.112 |
| NE | -0.004 |
| W | -0.254 |
| N | -0.261 |
| NW | -0.327 |

Diễn giải hướng: source→target SE nghĩa source nằm phía tây-bắc của target. Một pressure signal ở source phía tây/tây-bắc chứa nhiều thông tin cho target phía đông/đông-nam, trong khi chiều ngược lại không những yếu mà partial correlation thường âm.

Pattern không do một năm duy nhất. Trong band 100–800 km ở +6h:

- pressure SE: `0.400, 0.505, 0.472, 0.437` cho 2015–2018;
- pressure E: `0.298–0.362`;
- pressure W: `-0.139` đến `-0.228`;
- pressure NW: `-0.229` đến `-0.343`.

Temperature, RH và wind speed cũng có value-add cao hơn theo E/SE, nhưng mức bất đối xứng nhỏ hơn pressure. Đây phù hợp với các hệ synoptic/front di chuyển có hướng, song topology trạm và địa hình vẫn có thể góp phần; không gọi đây là causal advection nếu chưa có kiểm định mô hình.

### 3.5 “Best neighbour lag” không phải travel time thuần

Nhiều pressure/temperature edge đạt partial value-add lớn nhất ở +24h. Điều này không có nghĩa front mất đúng 24h để đi giữa hai trạm: khi lead tăng, target-current persistence yếu đi nên neighbour có nhiều phần variance để bổ sung hơn. Vì vậy file `directed_best_lag_edges.csv` nên được đọc là **horizon source station hữu ích nhất ngoài target persistence**, không phải tốc độ truyền trực tiếp.

Travel-time inference cần condition thêm trajectory, wind field và compare lag ở cùng target uncertainty. Đây là một giới hạn quan trọng để tránh overclaim từ biểu đồ lag.

## 4. Các “quy luật ẩn” cô đọng

1. **Hai tầng động lực:** regional slow manifold + local fast noise.
2. **Recent-24h sufficiency:** cho day 1, 24h gần nhất thường giàu thông tin hơn full 72h; 48h trước chủ yếu giúp nhận dạng regime.
3. **Variable clocks:** pressure/temperature/RH chậm; wind/PBL/rain nhanh.
4. **Extreme-regime stickiness:** quintile cực trị bền hơn quintile giữa.
5. **Analog horizon collapse:** analog retrieval hữu ích ngày 1, gần vô nghĩa ngày 3.
6. **Latitude-conditioned memory:** phía bắc cần decay nhanh/uncertainty cao hơn phía nam.
7. **Season-conditioned predictability:** summer slow/stable, spring fast/khó dự báo xa.
8. **Feature-specific spatial radius:** pressure global, temperature regional, wind/PBL local.
9. **Anisotropic directed graph:** west/northwest source → east/southeast target đặc biệt quan trọng cho pressure.
10. **Trend hierarchy:** pressure/temperature slope có value-add nhiều ngày; RH/wind chủ yếu ngày đầu; PBL/rain slope gần như không hữu ích.

## 5. Hàm ý kiến trúc mô hình

Một thiết kế phù hợp với EDA này nên có:

- **Regional latent encoder:** vài spatial modes cho pressure/temperature/RH.
- **Local residual branch:** receptive field ngắn và uncertainty lớn hơn, nhất là wind/PBL/rain.
- **Multi-resolution history:** last 6–24h, previous day và old 24–72h đi qua nhánh riêng; không average đồng đều.
- **Station/latitude-conditioned decay:** temporal transition phụ thuộc station embedding hoặc geographic covariates.
- **Season-gated dynamics:** mùa điều khiển transition/noise scale, không chỉ climatological bias.
- **Feature-specific graph:** radius và edge weighting khác nhau theo biến.
- **Directed anisotropic message passing:** source→target edge phụ thuộc displacement/bearing và horizon; adjacency đối xứng sẽ làm mất pattern pressure rõ nhất.
- **Trend channels có chọn lọc:** extrapolate pressure/temperature, regularize mạnh slope của PBL/rain.
- **Horizon-dependent uncertainty:** day 3 phải dựa nhiều hơn vào regional/climatological state; analog/local history không đủ ràng buộc.

## 6. Giới hạn

- Cửa sổ chồng lấn không độc lập; các con số là mô tả effect size, không phải IID p-value.
- Climatology và PCA fit trên toàn bộ dữ liệu vì mục tiêu là EDA toàn bộ. Khi đánh giá forecast, mọi transform phải fit train-only.
- Partial correlation control target hiện tại nhưng chưa loại mọi common driver, địa hình hoặc sampling geometry.
- Directional network dùng geographic bearing, chưa condition trực tiếp wind direction theo từng timestamp.
- Spatial PC là linear modes; cấu trúc thực có thể phi tuyến và thay đổi theo mùa.
- Clock timezone không được khẳng định từ metadata hiện có.
- Analog score đo “hai history giống nhau có dẫn đến hai future giống nhau” chứ chưa phải lỗi của một analog forecasting algorithm hoàn chỉnh.

## 7. Artifacts chính

- `within_station_hidden_structure.png`: memory, history relevance, analog và regional/local decomposition.
- `station_persistence_geography.png`: gradient theo vĩ độ.
- `spatial_coherence_by_distance.png`: correlation length theo biến.
- `spatial_pca_structure.png`: low-rank regional modes.
- `directed_cross_station_dynamics.png`: directional partial correlation và best-value-add horizon.
- `within_station_persistence.csv`, `history_offset_relevance.csv`, `weather_trend_value_add.csv`.
- `multivariate_analog_predictability.csv`, `station_regime_transitions.csv`.
- `cross_station_spatial_summary.csv`, `directed_best_lag_edges.csv`, `yearly_directional_robustness.csv`.
- `summary.json`: bản machine-readable.
