"""Build the five reproducible benchmark EDA notebooks.

The notebooks intentionally keep dataset-specific loading code visible.  They
share the same scientific questions: data quality, persistence versus future
change, spike onset, spatial context, seasonality/regime, and temporal shift.
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "pollution-venv",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, NOTEBOOK_DIR / name)


def cross_benchmark_cells(dataset_labels: list[str]) -> list:
    labels = repr(dataset_labels)
    return [
        md(
            """
            ## Standardized cross-benchmark diagnostics

            Các bảng dưới đây dùng cùng định nghĩa horizon, station-specific
            train p90, train IQR, chronological split và mask-aware correlation.
            Chúng giúp phân biệt quan sát lặp lại giữa benchmark với đặc điểm chỉ
            có ở dataset này. Kết quả là mô tả dữ liệu, không phải causal claim
            hay quyết định kiến trúc.

            `spatial_gap_partial_current` quan trọng hơn raw spatial-gap
            correlation vì đã giảm tương quan cơ học do cả spatial gap và target
            change cùng chứa giá trị PM hiện tại.
            """
        ),
        code(
            f"""
from IPython.display import Image

cross_dir = ROOT / "artifacts/cross_dataset_eda"
labels = {labels}
overview_cross = pd.read_csv(cross_dir / "dataset_overview.csv")
horizon_cross = pd.read_csv(cross_dir / "horizon_signals.csv")
events_cross = pd.read_csv(cross_dir / "events_missingness_shift.csv")
edge_cross = pd.read_csv(cross_dir / "edge_lead_lag.csv")
distance_cross = pd.read_csv(cross_dir / "distance_decay_summary.csv")
wind_cross = pd.read_csv(cross_dir / "wind_alignment_replication.csv")
coupling_cross = pd.read_csv(cross_dir / "pollutant_coupling.csv")
meteo_cross = pd.read_csv(cross_dir / "meteorology_precursors.csv")

display(Image(filename=str(cross_dir / "cross_dataset_main.png"), width=1100))
display(overview_cross.query("dataset in @labels").round(3))
display(horizon_cross.query("dataset in @labels").round(3))
display(events_cross.query("dataset in @labels and current_over_p90_bin != current_over_p90_bin").round(3))
display(events_cross.query("dataset in @labels and current_over_p90_bin == current_over_p90_bin")[[
    "dataset", "current_over_p90_bin", "n", "onset_24h_rate"
]].round(3))
display(edge_cross.query("dataset in @labels").round(3))
display(distance_cross.query("dataset in @labels").round(3))
display(wind_cross.query("dataset in @labels").round(3))
display(coupling_cross.query("dataset in @labels").round(3))
display(meteo_cross.query("dataset in @labels").round(3))
"""
        ),
        md(
            """
            Báo cáo diễn giải đầy đủ nằm tại
            `artifacts/cross_dataset_eda/OBSERVATIONS.md`. Các ô trống có nghĩa là
            phép đo không hợp lệ hoặc không khả dụng trên release này; không được
            diễn giải thành “không có hiệu ứng”.
            """
        ),
    ]


COMMON_IMPORTS = """
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)
RNG = np.random.default_rng(42)
"""


def uci_notebook() -> list:
    return [
        md(
            """
            # 01 — UCI Beijing Multi-Site: predictive EDA

            **Nguồn:** [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data),
            12 trạm, dữ liệu theo giờ 2013–2017.

            Notebook trả lời sáu câu hỏi dùng chung cho cả benchmark:

            1. Dữ liệu thiếu và lệch phân phối ở đâu?
            2. Future level khác future change như thế nào?
            3. Có precursor cho spike onset không?
            4. Trạng thái trạm khác có ích hơn lịch sử riêng không?
            5. Khí tượng/calendar tạo regime nào?
            6. Validation và test có cùng độ khó không?
            """
        ),
        code(
            COMMON_IMPORTS
            + """
from benchmark_eda.data import load_raw_frames, RAW_FEATURES

raw_dir = ROOT / "data/raw/PRSA_Data_20130301-20170228"
df = load_raw_frames(raw_dir).sort_values(["station", "timestamp"]).copy()
group = df.groupby("station", sort=False)
for lag in [1, 3, 6, 12, 24]:
    df[f"PM_trend_{lag}h"] = df["PM2.5"] - group["PM2.5"].shift(lag)
for horizon in range(1, 7):
    df[f"PM_future_{horizon}h"] = group["PM2.5"].shift(-horizon)
    df[f"PM_delta_{horizon}h"] = df[f"PM_future_{horizon}h"] - df["PM2.5"]

overview = pd.Series({
    "rows": len(df), "stations": df.station.nunique(),
    "start": df.timestamp.min(), "end": df.timestamp.max(),
    "PM2.5 missing %": df["PM2.5"].isna().mean() * 100,
    "duplicate station-hours": df.duplicated(["station", "timestamp"]).sum(),
})
overview
"""
        ),
        md("## Data quality, heterogeneity và seasonality"),
        code(
            """
missing = df[RAW_FEATURES].isna().mean().mul(100).sort_values(ascending=False)
station_pm = df.groupby("station")["PM2.5"].agg(
    count="count", mean="mean", median="median",
    p90=lambda x: x.quantile(.9), p99=lambda x: x.quantile(.99)
).sort_values("mean", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
missing.plot.bar(ax=axes[0], title="Missingness theo feature (%)")
station_pm["mean"].plot.bar(ax=axes[1], title="PM2.5 mean theo station")
df.assign(month=df.timestamp.dt.month).groupby("month")["PM2.5"].median().plot(
    marker="o", ax=axes[2], title="Median PM2.5 theo tháng"
)
plt.tight_layout()
display(missing.round(2).to_frame("missing_pct"))
display(station_pm.round(1))
"""
        ),
        md("## Persistence không đồng nghĩa với dự báo được transition"),
        code(
            """
predictors = RAW_FEATURES + [f"PM_trend_{lag}h" for lag in [1, 3, 6, 12, 24]]
level_corr, delta_corr = {}, {}
for horizon in [1, 3, 6]:
    cols = predictors + [f"PM_future_{horizon}h", f"PM_delta_{horizon}h"]
    corr = df[cols].corr(method="spearman")
    level_corr[horizon] = corr[f"PM_future_{horizon}h"].loc[predictors]
    delta_corr[horizon] = corr[f"PM_delta_{horizon}h"].loc[predictors]
level_corr, delta_corr = pd.DataFrame(level_corr), pd.DataFrame(delta_corr)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
sns.heatmap(level_corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axes[0])
axes[0].set_title("Spearman với future PM2.5 level")
sns.heatmap(delta_corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axes[1])
axes[1].set_title("Spearman với future PM2.5 change")
plt.tight_layout()
display(delta_corr.reindex(delta_corr[6].abs().sort_values(ascending=False).index).round(3))
"""
        ),
        md("## Spike onset: kiểm soát theo mức PM hiện tại"),
        code(
            """
timestamps = np.sort(df.timestamp.unique())
train_end = timestamps[int(len(timestamps) * .70)]
threshold = df.loc[df.timestamp < train_end, "PM2.5"].quantile(.90)
future_cols = [f"PM_future_{h}h" for h in range(1, 7)]
eligible = df["PM2.5"].lt(threshold) & df[future_cols].notna().all(axis=1)
df["onset_next6"] = df[future_cols].ge(threshold).any(axis=1) & df["PM2.5"].lt(threshold)
relative_bins = [0, .25, .50, .75, 1.0]
df["relative_level_bin"] = pd.cut(
    df["PM2.5"] / threshold, relative_bins, right=False
)
onset_by_level = (
    df.loc[eligible].groupby("relative_level_bin", observed=True)["onset_next6"]
    .agg(["count", "sum", "mean"])
)

episode_lengths = []
for _, station_frame in df.groupby("station"):
    flag = station_frame["PM2.5"].ge(threshold).fillna(False).to_numpy()
    starts = np.where(flag & ~np.r_[False, flag[:-1]])[0]
    ends = np.where(flag & ~np.r_[flag[1:], False])[0]
    episode_lengths.extend((ends - starts + 1).tolist())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
onset_by_level["mean"].mul(100).plot.bar(ax=axes[0], color="#D95858")
axes[0].set(title="P(onset 6h | current / train-p90)", ylabel="%")
sns.histplot(episode_lengths, bins=range(1, 49), ax=axes[1])
axes[1].set(title="Spike duration (zoom 1–48h)", xlabel="hours", xlim=(1, 48))
plt.tight_layout()
display(onset_by_level.assign(rate_pct=onset_by_level["mean"] * 100).round(3))
display(pd.Series(episode_lengths).quantile([.5, .75, .9, .95, 1]).to_frame("hours"))
"""
        ),
        md("## Spatial context và lead–lag"),
        code(
            """
wide_pm = df.pivot(index="timestamp", columns="station", values="PM2.5").sort_index()
valid_count = wide_pm.notna().sum(axis=1)
city_sum = wide_pm.sum(axis=1, min_count=2)
spatial_rows = []
for station in wide_pm:
    own = wide_pm[station]
    other = (city_sum - own) / (valid_count - 1)
    for horizon in [1, 3, 6]:
        future_delta = own.shift(-horizon) - own
        signals = {
            "own level": own, "own trend 1h": own.diff(),
            "other-station level": other,
            "other-station trend 1h": other.diff(),
            "spatial gap (other-own)": other - own,
        }
        for name, values in signals.items():
            spatial_rows.append(
                (station, horizon, name, values.corr(future_delta, method="spearman"))
            )
spatial_signal = pd.DataFrame(
    spatial_rows, columns=["station", "horizon", "signal", "rho"]
)
spatial_summary = spatial_signal.groupby(["horizon", "signal"]).rho.agg(["mean", "std"])

delta_wide = wide_pm.diff()
best_leads = []
for source in delta_wide:
    for target in delta_wide:
        if source == target:
            continue
        correlations = {
            lead: delta_wide[source].corr(delta_wide[target].shift(-lead), method="spearman")
            for lead in range(7)
        }
        best_leads.append(max(correlations, key=lambda lag: abs(correlations[lag])))
display(spatial_summary.round(3))
display(pd.Series(best_leads).value_counts().sort_index().to_frame("station_pairs"))
"""
        ),
        md("## Regime khí tượng và distribution shift"),
        code(
            """
df["month"] = df.timestamp.dt.month
df["hour"] = df.timestamp.dt.hour
onset_month = df.loc[eligible].groupby("month")["onset_next6"].mean() * 100
onset_hour = df.loc[eligible].groupby("hour")["onset_next6"].mean() * 100
onset_wind = (
    df.loc[eligible].groupby("wd")["onset_next6"].agg(["count", "mean"])
    .query("count > 1000").sort_values("mean", ascending=False)
)

val_end = timestamps[int(len(timestamps) * .85)]
df["split"] = np.where(
    df.timestamp < train_end, "train",
    np.where(df.timestamp < val_end, "val", "test")
)
shift = df.groupby("split").agg(
    PM_mean=("PM2.5", "mean"),
    PM_p99=("PM2.5", lambda x: x.quantile(.99)),
    delta6_abs_p90=("PM_delta_6h", lambda x: x.abs().quantile(.90)),
)
shift["onset6_pct"] = (
    df.loc[eligible].groupby("split")["onset_next6"].mean() * 100
)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
onset_month.plot(marker="o", ax=axes[0], title="Onset theo tháng (%)")
onset_hour.plot(marker="o", ax=axes[1], title="Onset theo giờ (%)")
onset_wind["mean"].mul(100).plot.bar(ax=axes[2], title="Onset theo hướng gió (%)")
plt.tight_layout()
display(onset_wind.assign(rate_pct=onset_wind["mean"] * 100).round(3))
display(shift.round(2))
"""
        ),
        code(
            """
from benchmark_eda.deep_eda import run_deep_eda
from IPython.display import Image

deep_dir = ROOT / "artifacts/deep_eda_uci"
deep_summary = run_deep_eda(raw_dir, ROOT / "data/metadata/uci_beijing_station_coords.csv", deep_dir)
regional = pd.read_csv(deep_dir / "regional_local_variance.csv")
pairwise = pd.read_csv(deep_dir / "pairwise_functional_graph.csv")
spikes = pd.read_csv(deep_dir / "spike_episodes.csv")
ridge = pd.read_csv(deep_dir / "diagnostic_ridge_ablation.csv")
spatial_gain = pd.read_csv(deep_dir / "spatial_gain_by_regime.csv")
wind_contrast = pd.read_csv(deep_dir / "wind_alignment_contrast.csv")
wind_robustness = pd.read_csv(deep_dir / "wind_alignment_robustness.csv")

display(Image(filename=str(deep_dir / "deep_eda_main.png"), width=1050))
display(regional.round(3))
display(wind_contrast.round(3))
display(wind_robustness.round(3))
display(ridge.round(3))
"""
        ),
        md(
            """
            ## Deep EDA sau khi khóa mô hình

            Phần này đi xa hơn correlation toàn cục và kiểm tra trực tiếp các
            giả thuyết kiến trúc. Split 70/15/15 vẫn theo thời gian; mọi threshold,
            imputation và scaler của diagnostic Ridge chỉ được fit trên train.

            - **Regional–local decomposition:** tách biến thiên của future change
              thành thành phần city-wide và residual riêng từng trạm.
            - **Functional graph:** so KNN địa lý với top-neighbour theo residual
              change, sau khi bỏ city-wide factor.
            - **Event structure:** phân biệt spike cô lập, cluster và regional.
            - **Wind-conditioned pair lag:** với mỗi directed geographic edge,
              so correlation source→target khi gió thổi cùng hướng và ngược hướng.
            - **Conditional incremental value:** Ridge chỉ đóng vai trò probe tuyến
              tính để hỏi spatial context có ích ở regime nào, không phải baseline
              cạnh tranh với neural model.
            """
        ),
        code(
            """
regional_share = regional.set_index("horizon").regional_share_of_component_variance
lag1 = wind_contrast.set_index("lag_h").loc[1]
knn_overlap = deep_summary["regional_functional_graph"][
    "geographic_knn_overlap_with_functional_top4_mean"
]
spike_share = deep_summary["spike_missingness"][
    "active_spike_hour_category_share"
]
peripheral_overlap = pd.Series(
    deep_summary["regional_functional_graph"]["geographic_knn_overlap_by_station"]
).sort_values()
top_gain = spatial_gain.sort_values("mean_gain", ascending=False).head(12)

display(Markdown(f'''
### Các quan sát mới

1. **Bài toán đổi bản chất theo horizon.** Regional share tăng từ
   **{regional_share.loc[1]:.1%} ở +1h** lên **{regional_share.loc[24]:.1%} ở +24h**.
   Vì vậy một spatial block giống nhau cho mọi horizon là giả định quá mạnh.
2. **Địa lý là prior tốt nhưng chưa đủ cho mọi trạm.** Mean overlap giữa geographic
   KNN và functional top-4 là **{knn_overlap:.1%}**; thấp nhất ở
   **{peripheral_overlap.index[0]} ({peripheral_overlap.iloc[0]:.0%})** và
   **{peripheral_overlap.index[1]} ({peripheral_overlap.iloc[1]:.0%})**.
3. **Gió không tạo một dynamic graph chung cho mọi lag.** Quan hệ mạnh nhất vẫn
   đồng thời ở hầu hết edge. Tuy nhiên tại lag +1h, aligned-minus-opposed rho là
   **{lag1['mean']:.3f}**, dương ở **{lag1['positive_pair_fraction']:.1%} edge**;
   hiệu ứng xuất hiện lại ở cả train/val/test, bốn mùa và hai mức wind speed.
4. **Spike gồm hai cơ chế.** Trong active spike-hours,
   **{spike_share['regional_6_12']:.1%}** là regional nhưng
   **{spike_share['isolated_1_2']:.1%}** chỉ xảy ra ở 1–2 trạm.
5. **Spatial value có điều kiện.** Probe tuyến tính cho thấy gain lớn nhất quanh
   +6h, khi spatial gap lớn, pollution cao, mùa đông và ở một số trạm ngoại vi.
   Đây là bằng chứng cho gating theo regime, không phải cho graph phức tạp hơn.
'''))
display(top_gain.round(3))
"""
        ),
        code(
            """
display(Markdown(f'''
## Kết luận và giả thuyết nghiên cứu

- Future level bị persistence chi phối (`rho(+1h)={level_corr.loc["PM2.5", 1]:.2f}`),
  còn future change yếu và đổi cơ chế theo horizon.
- Dài hạn chủ yếu là regional factor; ngắn hạn mới chứa directional transport
  residual. Kiến trúc nên tách hai cơ chế này thay vì buộc một graph xử lý cả hai.
- Candidate 1: **horizon-conditioned regional/local mixture** — trọng số regional
  tăng theo horizon, local branch giữ transition ngắn hạn.
- Candidate 2: **sparse one-hour wind transport residual** — wind chỉ điều chỉnh
  residual source→target ở lag ngắn, không thay toàn bộ fixed graph.
- Candidate 3: **regime-gated spatial correction** — kích hoạt mạnh hơn khi spatial
  disagreement lớn, mùa đông/high pollution hoặc ở trạm ngoại vi.
- Candidate 4: **dual event objective** — dự báo riêng xác suất isolated/local onset
  và regional episode, bên cạnh concentration MAE.

Các candidate trên là giả thuyết sinh từ dữ liệu, chưa phải novelty claim. Cần
kiểm tra trên benchmark khác và đối chiếu literature trước khi chọn đóng góp chính.
'''))
"""
        ),
    ]


def kdd_notebook() -> list:
    return [
        md(
            """
            # 02 — Beijing/London KDD Cup 2018: predictive EDA

            **Nguồn:** [KDD Cup 2018](https://www.kdd.org/kdd2018/kdd-cup) và bản
            lưu trữ [Zenodo DOI 10.5281/zenodo.4656719](https://doi.org/10.5281/zenodo.4656719),
            giấy phép CC BY 4.0.

            Dữ liệu gồm 270 chuỗi pollutant theo giờ, 35 trạm Beijing và 24 trạm
            London. Bản TSF này giữ missing value nhưng không chứa meteorology,
            nên notebook tập trung vào persistence, missingness, spike và spatial context.
            """
        ),
        code(
            COMMON_IMPORTS
            + """
tsf_path = ROOT / "data/benchmarks/beijing_kdd/kdd_cup_2018_dataset_with_missing_values.tsf"
records = []
in_data = False
with tsf_path.open() as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        name, city, station, pollutant, start, raw_values = line.split(":", 5)
        values = np.array(
            [np.nan if value == "?" else float(value) for value in raw_values.split(",")],
            dtype=np.float32,
        )
        records.append({
            "series": name, "city": city, "station": station,
            "pollutant": pollutant,
            "start": pd.to_datetime(start, format="%Y-%m-%d %H-%M-%S"),
            "values": values, "length": len(values),
            "missing_pct": np.isnan(values).mean() * 100,
        })
catalog = pd.DataFrame(records)

def make_panel(city, pollutant):
    series = []
    for row in catalog.query("city == @city and pollutant == @pollutant").itertuples():
        index = pd.date_range(row.start, periods=row.length, freq="h")
        series.append(pd.Series(row.values, index=index, name=row.station))
    return pd.concat(series, axis=1).sort_index()

overview = catalog.groupby(["city", "pollutant"]).agg(
    series=("series", "count"), median_length=("length", "median"),
    missing_pct=("missing_pct", "mean")
)
display(overview.round(2))
"""
        ),
        md("## Missingness là cấu trúc, không phải nhiễu ngẫu nhiên"),
        code(
            """
def missing_run_lengths(values):
    flag = np.isnan(values)
    padded = np.r_[False, flag, False].astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return ends - starts

gap_rows = []
for row in catalog.itertuples():
    gaps = missing_run_lengths(row.values)
    gap_rows.append({
        "city": row.city, "pollutant": row.pollutant,
        "missing_pct": row.missing_pct,
        "gap_p90_h": np.quantile(gaps, .9) if len(gaps) else 0,
        "gap_max_h": gaps.max() if len(gaps) else 0,
    })
gap_summary = pd.DataFrame(gap_rows).groupby(["city", "pollutant"]).median()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
overview["missing_pct"].unstack(0).plot.bar(ax=axes[0], title="Mean missingness (%)")
gap_summary["gap_max_h"].unstack(0).plot.bar(ax=axes[1], title="Median max gap (hours)")
plt.tight_layout()
display(gap_summary.round(1))
"""
        ),
        md("## Future change và spatial context theo city"),
        code(
            """
def future_spatial_summary(panel, horizons):
    valid_count = panel.notna().sum(axis=1)
    total = panel.sum(axis=1, min_count=2)
    rows = []
    for station in panel:
        own = panel[station]
        other = (total - own) / (valid_count - 1)
        for horizon in horizons:
            future_delta = own.shift(-horizon) - own
            signals = {
                "own level": own,
                "own trend 1h": own.diff(),
                "other trend 1h": other.diff(),
                "spatial gap": other - own,
            }
            for signal, values in signals.items():
                rows.append((station, horizon, signal,
                             values.corr(future_delta, method="spearman")))
    return pd.DataFrame(rows, columns=["station", "horizon", "signal", "rho"])

panels, spatial_parts = {}, []
for city in ["Beijing", "London"]:
    panel = make_panel(city, "PM2.5")
    panels[city] = panel
    part = future_spatial_summary(panel, [1, 6, 24, 48])
    part["city"] = city
    spatial_parts.append(part)
spatial = pd.concat(spatial_parts)
spatial_summary = spatial.groupby(["city", "horizon", "signal"]).rho.mean()
display(spatial_summary.unstack("signal").round(3))

coords = pd.read_csv(
    ROOT / "data/benchmarks/beijing_kdd/beijing_station_coords.csv"
).set_index("station")
beijing_panel = panels["Beijing"]
shared = [station for station in beijing_panel if station in coords.index]
coord_radians = np.radians(coords.loc[shared, ["latitude", "longitude"]].to_numpy())
dlat = coord_radians[:, None, 0] - coord_radians[None, :, 0]
dlon = coord_radians[:, None, 1] - coord_radians[None, :, 1]
a = np.sin(dlat / 2) ** 2 + np.cos(coord_radians[:, None, 0]) * np.cos(
    coord_radians[None, :, 0]
) * np.sin(dlon / 2) ** 2
distance = 6371 * 2 * np.arcsin(np.sqrt(a))
pm_corr = beijing_panel[shared].corr(method="spearman").to_numpy()
upper = np.triu_indices(len(shared), 1)
distance_decay = pd.DataFrame({
    "distance_km": distance[upper], "PM_correlation": pm_corr[upper]
})
distance_decay["distance_bin"] = pd.cut(
    distance_decay.distance_km, [0, 10, 20, 40, 80, np.inf]
)
display(distance_decay.groupby("distance_bin", observed=True).PM_correlation
        .agg(["count", "median", "mean"]).round(3))
"""
        ),
        md("## Spike onset, seasonality và city shift"),
        code(
            """
onset_rows, monthly_parts, shift_rows = [], [], []
for city, panel in panels.items():
    cut = int(len(panel) * .70)
    threshold = np.nanquantile(panel.iloc[:cut].to_numpy(), .90)
    future_max = pd.DataFrame(
        np.nanmax(np.stack([panel.shift(-lead).to_numpy()
                           for lead in range(1, 25)]), axis=0),
        index=panel.index, columns=panel.columns,
    )
    eligible = panel.lt(threshold) & panel.notna() & future_max.notna()
    onset = future_max.ge(threshold) & panel.lt(threshold)
    ratio = panel / threshold
    for low, high in zip([0, .25, .5, .75], [.25, .5, .75, 1]):
        mask = eligible & ratio.ge(low) & ratio.lt(high)
        onset_rows.append({
            "city": city, "current/train_p90": f"{low:.2f}–{high:.2f}",
            "observations": int(mask.sum().sum()),
            "onset_24h_pct": onset.where(mask).stack().mean() * 100,
        })
    city_mean = panel.mean(axis=1)
    monthly = city_mean.groupby(city_mean.index.month).median()
    monthly.name = city
    monthly_parts.append(monthly)
    for label, subset in {
        "train70": panel.iloc[:cut],
        "test30": panel.iloc[cut:],
    }.items():
        delta24 = subset.shift(-24) - subset
        shift_rows.append({
            "city": city, "split": label,
            "PM_mean": subset.stack().mean(),
            "PM_p99": subset.stack().quantile(.99),
            "|delta24|_p90": delta24.stack().abs().quantile(.90),
        })
onset_table = pd.DataFrame(onset_rows)
monthly = pd.concat(monthly_parts, axis=1)
shift = pd.DataFrame(shift_rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.barplot(data=onset_table, x="current/train_p90", y="onset_24h_pct",
            hue="city", ax=axes[0])
axes[0].set_title("Spike onset trong 24h")
monthly.plot(marker="o", ax=axes[1], title="Median city PM2.5 theo tháng")
plt.tight_layout()
display(onset_table.round(2))
display(shift.round(2))
"""
        ),
        code(
            """
comparison = spatial_summary.unstack("signal")
display(Markdown(f'''
## Kết luận có thể dùng chung

- Cả hai city đều có missing gap dài; forward-fill vô điều kiện sẽ làm giả
  persistence và graph correlation.
- Ở Beijing +24h, other-station trend có `rho={comparison.loc[("Beijing", 24), "other trend 1h"]:.3f}`;
  spatial gap có `rho={comparison.loc[("Beijing", 24), "spatial gap"]:.3f}`.
- KDD hỗ trợ đánh giá spatial context và forecast dài 24–48h, nhưng bản TSF
  không đủ để nghiên cứu tác động meteorology.
- Beijing và London khác scale, pollutant coverage và seasonality; không nên
  dùng một threshold tuyệt đối hoặc một scaler chung cho hai city.
'''))
"""
        ),
    ]


def knowair_notebook() -> list:
    return [
        md(
            """
            # 03 — KnowAir: predictive EDA

            **Nguồn chính thức:** [PM2.5-GNN repository](https://github.com/shuowang-ai/PM2.5-GNN),
            184 thành phố, chu kỳ 3 giờ, 2015–2018. Tensor gồm 17 biến khí tượng
            và PM2.5.

            Đây là benchmark phù hợp nhất để kiểm tra đồng thời temporal dynamics,
            city graph, khoảng cách địa lý và meteorological precursors.
            """
        ),
        code(
            COMMON_IMPORTS
            + """
FEATURES = [
    "100m_u_wind", "100m_v_wind", "2m_dewpoint", "2m_temperature",
    "boundary_layer_height", "k_index", "relative_humidity_950",
    "relative_humidity_975", "specific_humidity_950", "surface_pressure",
    "temperature_925", "temperature_950", "total_precipitation",
    "u_wind_950", "v_wind_950", "vertical_velocity_950",
    "vorticity_950", "PM2.5",
]
array = np.load(ROOT / "data/benchmarks/knowair/KnowAir.npy", mmap_mode="r")
timestamps = pd.date_range("2015-01-01", periods=array.shape[0], freq="3h")
cities = pd.read_csv(
    ROOT / "data/benchmarks/knowair/city.txt", sep=r"\\s+",
    names=["id", "city", "lon", "lat"]
)
pm = np.asarray(array[:, :, -1], dtype=np.float32)
overview = pd.Series({
    "shape": str(array.shape), "start": timestamps.min(), "end": timestamps.max(),
    "cities": array.shape[1], "features": array.shape[2],
    "nonfinite values": int((~np.isfinite(array)).sum()),
    "PM2.5 p50": np.nanquantile(pm, .5),
    "PM2.5 p90": np.nanquantile(pm, .9),
    "PM2.5 p99": np.nanquantile(pm, .99),
})
overview
"""
        ),
        md("## Spatial heterogeneity và seasonality"),
        code(
            """
pm_frame = pd.DataFrame(pm, index=timestamps, columns=cities.city)
city_stats = pm_frame.agg(["mean", "median", "std"]).T
city_stats["p90"] = pm_frame.quantile(.9)
city_stats["p99"] = pm_frame.quantile(.99)
monthly = pm_frame.groupby(pm_frame.index.month).median().median(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
sns.histplot(pm_frame.stack(), bins=80, ax=axes[0])
axes[0].set(xlim=(0, pm_frame.stack().quantile(.995)), title="PM2.5 distribution")
city_stats["mean"].sort_values(ascending=False).head(30).plot.bar(
    ax=axes[1], title="30 city có mean PM2.5 cao nhất"
)
monthly.plot(marker="o", ax=axes[2], title="Median PM2.5 theo tháng")
plt.tight_layout()
display(city_stats.sort_values("mean", ascending=False).head(20).round(1))
"""
        ),
        md("## Persistence, future change và meteorological precursors"),
        code(
            """
sample_t = RNG.choice(array.shape[0] - 8, size=min(2500, array.shape[0] - 8), replace=False)
sample_c = RNG.choice(array.shape[1], size=min(100, array.shape[1]), replace=False)
rows = []
for horizon_step in [1, 2, 8]:
    target_delta = (
        np.asarray(array[sample_t + horizon_step, :, -1])[:, sample_c]
        - np.asarray(array[sample_t, :, -1])[:, sample_c]
    ).ravel()
    feature_matrix = np.asarray(array[sample_t, :, :])[:, sample_c, :].reshape(-1, array.shape[-1])
    sampled = pd.DataFrame(feature_matrix, columns=FEATURES)
    sampled["future_delta"] = target_delta
    corr = sampled.corr(method="spearman")["future_delta"].drop("future_delta")
    for feature, rho in corr.items():
        rows.append((horizon_step * 3, feature, rho))
met_corr = pd.DataFrame(rows, columns=["horizon_h", "feature", "rho"])
top_met = (
    met_corr.assign(abs_rho=lambda x: x.rho.abs())
    .sort_values(["horizon_h", "abs_rho"], ascending=[True, False])
    .groupby("horizon_h").head(8)
)
display(top_met.pivot(index="feature", columns="horizon_h", values="rho").round(3))
"""
        ),
        md("## City context, spatial gap và distance decay"),
        code(
            """
city_mean = pm_frame.mean(axis=1)
rows = []
for city in pm_frame:
    own = pm_frame[city]
    other = (pm_frame.sum(axis=1) - own) / (pm_frame.shape[1] - 1)
    for horizon in [1, 2, 8]:
        delta = own.shift(-horizon) - own
        for signal, values in {
            "own trend 3h": own.diff(),
            "other-city trend 3h": other.diff(),
            "spatial gap": other - own,
            "own level": own,
        }.items():
            rows.append((city, horizon * 3, signal,
                         values.corr(delta, method="spearman")))
spatial = pd.DataFrame(rows, columns=["city", "horizon_h", "signal", "rho"])
spatial_summary = spatial.groupby(["horizon_h", "signal"]).rho.agg(["mean", "std"])

coords = np.radians(cities[["lat", "lon"]].to_numpy())
dlat = coords[:, None, 0] - coords[None, :, 0]
dlon = coords[:, None, 1] - coords[None, :, 1]
a = np.sin(dlat / 2) ** 2 + np.cos(coords[:, None, 0]) * np.cos(coords[None, :, 0]) * np.sin(dlon / 2) ** 2
distance = 6371 * 2 * np.arcsin(np.sqrt(a))
corr_matrix = pm_frame.corr(method="spearman").to_numpy()
upper = np.triu_indices(len(cities), 1)
distance_relation = pd.DataFrame({
    "distance_km": distance[upper], "PM_correlation": corr_matrix[upper]
})
distance_relation["distance_bin"] = pd.cut(
    distance_relation.distance_km, [0, 100, 300, 600, 1000, 2000, np.inf]
)

nearest_10 = np.argsort(distance, axis=1)[:, 1:11]
local_pm = np.nanmean(pm[:, nearest_10], axis=2)
local_rows = []
for city_idx, city in enumerate(pm_frame.columns):
    own = pm_frame[city]
    neighbour = pd.Series(local_pm[:, city_idx], index=pm_frame.index)
    for horizon in [1, 2, 8]:
        delta = own.shift(-horizon) - own
        for signal, values in {
            "own trend 3h": own.diff(),
            "local-neighbour trend 3h": neighbour.diff(),
            "local spatial gap": neighbour - own,
            "own level": own,
        }.items():
            local_rows.append(
                (city, horizon * 3, signal,
                 values.corr(delta, method="spearman"))
            )
local_spatial = pd.DataFrame(
    local_rows, columns=["city", "horizon_h", "signal", "rho"]
)
local_spatial_summary = local_spatial.groupby(
    ["horizon_h", "signal"]
).rho.agg(["mean", "std"])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
local_spatial.groupby(["horizon_h", "signal"]).rho.mean().unstack().plot.bar(ax=axes[0])
axes[0].set(title="Signals với future ΔPM2.5", ylabel="Spearman")
distance_relation.groupby("distance_bin", observed=True).PM_correlation.median().plot.bar(ax=axes[1])
axes[1].set(title="Distance decay của PM2.5 correlation", ylabel="Median Spearman")
plt.tight_layout()
display(spatial_summary.round(3).rename_axis(index={None: "global context"}))
display(local_spatial_summary.round(3))
display(distance_relation.groupby("distance_bin", observed=True).PM_correlation.agg(["count", "median", "mean"]).round(3))
"""
        ),
        md("## Spike onset và temporal shift"),
        code(
            """
train_cut = len(pm_frame) // 2
threshold = np.nanquantile(pm_frame.iloc[:train_cut].to_numpy(), .90)
future_max = pd.DataFrame(
    np.nanmax(np.stack([pm_frame.shift(-lead).to_numpy()
                       for lead in range(1, 9)]), axis=0),
    index=pm_frame.index, columns=pm_frame.columns,
)
eligible = pm_frame.lt(threshold) & future_max.notna()
onset = future_max.ge(threshold) & pm_frame.lt(threshold)
ratio = pm_frame / threshold
onset_rows = []
for low, high in zip([0, .25, .5, .75], [.25, .5, .75, 1]):
    mask = eligible & ratio.ge(low) & ratio.lt(high)
    onset_rows.append({
        "current/train_p90": f"{low:.2f}–{high:.2f}",
        "observations": int(mask.sum().sum()),
        "onset_24h_pct": onset.where(mask).stack().mean() * 100,
    })
onset_table = pd.DataFrame(onset_rows)

yearly = []
for year, values in pm_frame.groupby(pm_frame.index.year):
    delta24 = values.shift(-8) - values
    yearly.append({
        "year": year, "mean": values.stack().mean(),
        "p90": values.stack().quantile(.9), "p99": values.stack().quantile(.99),
        "|delta24|_p90": delta24.stack().abs().quantile(.9),
    })
display(onset_table.round(2))
display(pd.DataFrame(yearly).set_index("year").round(2))
"""
        ),
        code(
            """
display(Markdown(f'''
## Kết luận có thể dùng cho model

- KnowAir không có missing value gốc trong tensor phát hành; kết quả trên benchmark
  này không kiểm chứng được robustness với sensor outage.
- Local-neighbour trend tại +3h đạt
  `rho={local_spatial_summary.loc[(3, "local-neighbour trend 3h"), "mean"]:.3f}`;
  local spatial gap tăng tới
  `rho={local_spatial_summary.loc[(24, "local spatial gap"), "mean"]:.3f}` ở +24h.
- PM correlation giảm theo khoảng cách nhưng không biến mất ở khoảng cách lớn:
  graph cần local edges và một city/national context, không chỉ một trong hai.
- Meteorological relations đổi theo horizon; wind vector, boundary-layer height
  và humidity nên được kiểm tra theo regime thay vì chỉ global correlation.
'''))
"""
        ),
    ]


def airformer_notebook() -> list:
    return [
        md(
            """
            # 04 — AirFormer official tiny release: schema and representativeness audit

            **Nguồn:** [AirFormer official repository](https://github.com/yoshall/AirFormer).

            Paper dùng hơn 500 GB dữ liệu từ 1.085 trạm trong bốn năm, nhưng tác giả
            chỉ công bố **20 train + 20 validation + 20 test instances**. Vì vậy
            notebook này không trình bày seasonality/spike frequency như bằng chứng
            population-level. Mục tiêu hợp lệ là audit tensor schema, spatial coverage,
            target dynamics và mức độ đại diện của tiny release.
            """
        ),
        code(
            COMMON_IMPORTS
            + """
import zipfile

data_dir = ROOT / "data/benchmarks/airformer/extracted"
npz_files = sorted(data_dir.rglob("*.npz"))
file_catalog = []
loaded = {}
for path in npz_files:
    item = np.load(path)
    loaded[path.stem] = {key: item[key] for key in item.files}
    for key in item.files:
        arr = item[key]
        file_catalog.append({
            "file": str(path.relative_to(data_dir)), "key": key,
            "shape": str(arr.shape), "dtype": str(arr.dtype),
            "nonfinite": int((~np.isfinite(arr)).sum()) if np.issubdtype(arr.dtype, np.number) else np.nan,
        })
catalog = pd.DataFrame(file_catalog)
display(catalog)
"""
        ),
        md("## X/Y semantics, horizon dynamics và split comparability"),
        code(
            """
split_rows, split_arrays = [], {}
for split in ["train", "val", "test"]:
    candidates = [p for p in npz_files if p.stem.lower() == split]
    if not candidates:
        continue
    item = np.load(candidates[0])
    x, y = item["x"], item["y"]
    split_arrays[split] = (x, y)
    split_rows.append({
        "split": split, "instances": len(x),
        "x_shape": str(x.shape), "y_shape": str(y.shape),
        "x_mean": float(np.nanmean(x)), "x_std": float(np.nanstd(x)),
        "y_mean": float(np.nanmean(y)), "y_std": float(np.nanstd(y)),
    })
split_table = pd.DataFrame(split_rows).set_index("split")
display(split_table.round(3))

fig, axes = plt.subplots(1, len(split_arrays), figsize=(5 * len(split_arrays), 4))
axes = np.atleast_1d(axes)
for ax, (split, (_, y)) in zip(axes, split_arrays.items()):
    sns.histplot(np.asarray(y).ravel()[::max(1, y.size // 100_000)], bins=60, ax=ax)
    ax.set_title(f"{split}: sampled y distribution")
plt.tight_layout()
"""
        ),
        md("## Persistence residual và spatial heterogeneity trong packaged windows"),
        code(
            """
dynamic_rows = []
for split, (x, y) in split_arrays.items():
    # AirFormer stores target as channel 0; x/y use [sample,time,node,channel].
    target_x = np.asarray(x[..., 0])
    target_y = np.asarray(y[..., 0])
    persistence = target_x[:, -1:, :]
    residual = target_y - persistence
    city_trend = target_x[:, -1, :].mean(axis=1) - target_x[:, -2, :].mean(axis=1)
    own_trend = target_x[:, -1, :] - target_x[:, -2, :]
    first_delta = target_y[:, 0, :] - target_x[:, -1, :]
    dynamic_rows.append({
        "split": split,
        "persistence_MAE_h1": float(np.nanmean(np.abs(first_delta))),
        "residual_p50": float(np.nanquantile(residual, .5)),
        "|residual|_p90": float(np.nanquantile(np.abs(residual), .9)),
        "|residual|_p99": float(np.nanquantile(np.abs(residual), .99)),
        "own_trend_vs_h1_delta": pd.Series(own_trend.ravel()).corr(
            pd.Series(first_delta.ravel()), method="spearman"
        ),
        "city_trend_vs_mean_h1_delta": pd.Series(city_trend).corr(
            pd.Series(first_delta.mean(axis=1)), method="spearman"
        ),
        "node_mean_std": float(np.nanstd(target_x.mean(axis=(0, 1)))),
    })
dynamic_table = pd.DataFrame(dynamic_rows).set_index("split")
display(dynamic_table.round(3))
"""
        ),
        md("## Spatial assets và limitation audit"),
        code(
            """
other_files = sorted(
    path for path in data_dir.rglob("*")
    if path.is_file() and path.suffix.lower() not in {".npz"}
)
assets = pd.DataFrame({
    "file": [str(path.relative_to(data_dir)) for path in other_files],
    "size_mb": [path.stat().st_size / 2**20 for path in other_files],
})
display(assets.round(3))

import pickle
with open(data_dir / "sensor_graph/adj_mx_air_tiny.pkl", "rb") as handle:
    graph_payload = pickle.load(handle, encoding="latin1")
adjacency = graph_payload[-1]
degree = (adjacency > 0).sum(axis=1)
graph_audit = pd.Series({
    "nodes": adjacency.shape[0],
    "symmetric": bool(np.allclose(adjacency, adjacency.T)),
    "density_pct": np.count_nonzero(adjacency) / adjacency.size * 100,
    "degree_min": degree.min(), "degree_median": np.median(degree),
    "degree_p90": np.quantile(degree, .9), "degree_max": degree.max(),
})
partition_rows = []
for mask_path in sorted(data_dir.glob("local_partition/*/mask.npy")):
    mask = np.load(mask_path)
    partition_rows.append({
        "partition": mask_path.parent.name, "mask_shape": str(mask.shape),
        "active_fraction_pct": mask.mean() * 100,
        "groups_per_node_median": np.median(mask.sum(axis=1)),
    })
display(graph_audit.to_frame("value"))
display(pd.DataFrame(partition_rows).set_index("partition").round(2))

display(Markdown(f'''
## Kết luận hợp lệ

- Tiny release xác nhận đúng bài toán nationwide tensor và cho phép kiểm tra
  shape, normalization, output horizon, persistence residual và spatial assets.
- Chỉ có **{sum(len(x) for x, _ in split_arrays.values())} windows**; các window
  có thể chồng lấn hoặc được chọn có chủ đích. Không được dùng notebook này để
  kết luận spike rate, seasonal regime hay distribution shift của dữ liệu bốn năm.
- Mọi kết luận kiến trúc từ AirFormer phải dựa vào các benchmark đầy đủ khác;
  tiny release chỉ là reproducibility sample, không phải benchmark EDA độc lập.
'''))
"""
        ),
    ]


def airqualitybench_notebook() -> list:
    return [
        md(
            """
            # 05 — AirQualityBench: global, mask-aware predictive EDA

            **Nguồn:** [official repository](https://github.com/Star-Learning/AirQualityBench)
            và [Hugging Face dataset](https://huggingface.co/datasets/xuxing123/aq_dataset),
            giấy phép MIT.

            Bộ dữ liệu gồm 3.720 trạm toàn cầu, sáu pollutant, dữ liệu theo giờ
            2021–2025 và giữ nguyên observation mask. EDA ưu tiên missingness,
            geographic heterogeneity, temporal shift và local-neighbour signal.
            """
        ),
        code(
            COMMON_IMPORTS
            + """
import h5py
import pickle
from scipy import sparse

DATA_DIR = ROOT / "data/benchmarks/airqualitybench"
POLLUTANTS = ["pm25", "pm10", "no2", "o3", "so2", "co"]
metadata = pd.read_csv(DATA_DIR / "selected_nodes_metadata.csv")
schema_rows = []
for year in range(2021, 2026):
    with h5py.File(DATA_DIR / f"aq_compact_{year}.h5", "r") as handle:
        schema_rows.append({
            "year": year,
            "values_shape": str(handle["values"].shape),
            "masks_shape": str(handle["masks"].shape),
            "values_dtype": str(handle["values"].dtype),
            "mask_dtype": str(handle["masks"].dtype),
        })
display(pd.DataFrame(schema_rows).set_index("year"))
display(metadata.head())
"""
        ),
        md("## Authentic missingness và geographic coverage"),
        code(
            """
year_rows, coverage_sum = [], np.zeros((len(metadata), len(POLLUTANTS)), dtype=np.int64)
hours_total = 0
sample_values = {pollutant: [] for pollutant in POLLUTANTS}
for year in range(2021, 2026):
    with h5py.File(DATA_DIR / f"aq_compact_{year}.h5", "r") as handle:
        values, masks = handle["values"], handle["masks"]
        year_count = np.zeros(len(POLLUTANTS), dtype=np.int64)
        year_sum = np.zeros(len(POLLUTANTS), dtype=np.float64)
        for start in range(0, values.shape[0], 168):
            stop = min(start + 168, values.shape[0])
            block_v = values[start:stop]
            block_m = masks[start:stop].astype(bool)
            coverage_sum += block_m.sum(axis=0)
            year_count += block_m.sum(axis=(0, 1))
            year_sum += np.where(block_m, block_v, 0).sum(axis=(0, 1))
            # One hourly slice per week is enough for robust quantile sketches.
            sampled_v, sampled_m = block_v[::168], block_m[::168]
            for idx, pollutant in enumerate(POLLUTANTS):
                sample_values[pollutant].append(sampled_v[..., idx][sampled_m[..., idx]])
        hours_total += values.shape[0]
        total_slots = values.shape[0] * values.shape[1]
        for idx, pollutant in enumerate(POLLUTANTS):
            year_rows.append({
                "year": year, "pollutant": pollutant,
                "coverage_pct": year_count[idx] / total_slots * 100,
                "mean": year_sum[idx] / max(year_count[idx], 1),
            })
year_stats = pd.DataFrame(year_rows)
coverage = pd.DataFrame(
    coverage_sum / hours_total * 100, columns=POLLUTANTS
)
quantiles = pd.DataFrame({
    pollutant: np.quantile(np.concatenate(parts), [.5, .9, .99])
    for pollutant, parts in sample_values.items()
}, index=["p50", "p90", "p99"]).T

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
year_stats.pivot(index="year", columns="pollutant", values="coverage_pct").plot(
    marker="o", ax=axes[0], title="Valid observation coverage (%)"
)
coverage.boxplot(ax=axes[1], rot=45)
axes[1].set(title="Coverage distribution giữa stations", ylabel="%")
plt.tight_layout()
display(year_stats.pivot(index="year", columns="pollutant", values="coverage_pct").round(2))
display(quantiles.round(2))
"""
        ),
        md("## Missing gap length: mask là một phần của forecasting problem"),
        code(
            """
with h5py.File(DATA_DIR / "aq_compact_2025.h5", "r") as handle:
    pm_mask_2025 = handle["masks"][:, :, 0].astype(bool)
station_coverage_2025 = pm_mask_2025.mean(axis=0)
selected = np.argsort(station_coverage_2025)[-250:]

def false_run_lengths(flag):
    padded = np.r_[True, flag, True].astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.where(edges == -1)[0], np.where(edges == 1)[0]
    return ends - starts

gap_rows = []
for station in selected:
    gaps = false_run_lengths(pm_mask_2025[:, station])
    gap_rows.append({
        "station": station, "coverage_pct": station_coverage_2025[station] * 100,
        "missing_gap_p90_h": np.quantile(gaps, .9) if len(gaps) else 0,
        "missing_gap_max_h": gaps.max() if len(gaps) else 0,
    })
gap_table = pd.DataFrame(gap_rows)
display(gap_table.describe(percentiles=[.5, .9, .95, .99]).round(1))
"""
        ),
        md("## Local-neighbour context với future change"),
        code(
            """
with open(DATA_DIR / "adj_mx_10.pkl", "rb") as handle:
    adjacency_payload = pickle.load(handle)
adjacency = sparse.csr_matrix(adjacency_payload["adj_mx"])

with h5py.File(DATA_DIR / "aq_compact_2025.h5", "r") as handle:
    pm_2025 = handle["values"][:, :, 0].astype(np.float32)
    mask_2025 = handle["masks"][:, :, 0].astype(bool)
pm_2025[~mask_2025] = np.nan
filled = np.nan_to_num(pm_2025, nan=0.0)
neighbour_sum = adjacency.dot(filled.T).T
neighbour_weight = adjacency.dot(mask_2025.astype(np.float32).T).T
neighbour_pm = neighbour_sum / np.where(neighbour_weight > 0, neighbour_weight, np.nan)

spatial_rows = []
for station in selected:
    own = pd.Series(pm_2025[:, station])
    neighbour = pd.Series(neighbour_pm[:, station])
    for horizon in [1, 6, 24]:
        delta = own.shift(-horizon) - own
        for signal, values in {
            "own trend 1h": own.diff(),
            "neighbour trend 1h": neighbour.diff(),
            "local spatial gap": neighbour - own,
            "own level": own,
        }.items():
            spatial_rows.append(
                (station, horizon, signal, values.corr(delta, method="spearman"))
            )
spatial = pd.DataFrame(spatial_rows, columns=["station", "horizon_h", "signal", "rho"])
spatial_summary = spatial.groupby(["horizon_h", "signal"]).rho.agg(["mean", "std"])
display(spatial_summary.round(3))
"""
        ),
        md("## Spike onset, pollutant coupling và year shift"),
        code(
            """
train_pm_sketch = np.concatenate(sample_values["pm25"][:3 * 53])
threshold = np.quantile(train_pm_sketch, .90)
selected_pm = pd.DataFrame(pm_2025[:, selected])
future_max = pd.DataFrame(
    np.nanmax(np.stack([selected_pm.shift(-lead).to_numpy()
                       for lead in range(1, 25)]), axis=0),
    index=selected_pm.index, columns=selected_pm.columns,
)
eligible = selected_pm.lt(threshold) & selected_pm.notna() & future_max.notna()
onset = future_max.ge(threshold) & selected_pm.lt(threshold)
ratio = selected_pm / threshold
onset_rows = []
for low, high in zip([0, .25, .5, .75], [.25, .5, .75, 1]):
    condition = eligible & ratio.ge(low) & ratio.lt(high)
    onset_rows.append({
        "current/train_p90": f"{low:.2f}–{high:.2f}",
        "observations": int(condition.sum().sum()),
        "onset_24h_pct": onset.where(condition).stack().mean() * 100,
    })
onset_table = pd.DataFrame(onset_rows)

year_mean = year_stats.pivot(index="year", columns="pollutant", values="mean")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.barplot(data=onset_table, x="current/train_p90", y="onset_24h_pct", ax=axes[0])
axes[0].set_title("PM2.5 onset trong 24h, high-coverage stations")
year_mean.plot(marker="o", ax=axes[1], title="Observed mean theo year")
plt.tight_layout()
display(onset_table.round(2))
display(year_mean.round(2))
"""
        ),
        code(
            """
display(Markdown(f'''
## Kết luận có thể dùng cho model

- Missingness khác mạnh theo pollutant, station và year; impute rồi bỏ mask sẽ
  thay đổi chính forecasting problem.
- Ngay cả trong 250 trạm PM2.5 có coverage cao nhất, missing gaps vẫn có tail dài;
  masked loss và observed-state embedding là bắt buộc.
- Local-neighbour trend tại +1h có `rho={spatial_summary.loc[(1, "neighbour trend 1h"), "mean"]:.3f}`;
  local spatial gap tại +24h có `rho={spatial_summary.loc[(24, "local spatial gap"), "mean"]:.3f}`.
- Global mean không phải spatial context hợp lý; graph phải local hoặc
  hierarchical theo geography/provider regime.
- Year shift và scale khác nhau giữa pollutant khiến một scaler/loss toàn cục dễ
  ưu tiên CO hoặc các station có coverage dày.
'''))
"""
        ),
    ]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_uci_beijing_multisite.ipynb": uci_notebook() + cross_benchmark_cells(["UCI-Beijing"]),
        "02_beijing_kdd_cup.ipynb": kdd_notebook() + cross_benchmark_cells(["KDD-Beijing", "KDD-London"]),
        "03_knowair.ipynb": knowair_notebook() + cross_benchmark_cells(["KnowAir"]),
        "04_airformer.ipynb": airformer_notebook() + cross_benchmark_cells(["AirFormer-tiny"]),
        "05_airqualitybench.ipynb": airqualitybench_notebook() + cross_benchmark_cells(["AirQualityBench"]),
    }
    for name, cells in notebooks.items():
        write_notebook(name, cells)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
