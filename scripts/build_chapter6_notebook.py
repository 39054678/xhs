from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "chapter06" / "6.1_2.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# 6.1 发布时间、节假日与发布地的探索性分析

本 Notebook 是旧版 `6.1_2.ipynb` 的可复现整理版。它直接读取仓库
`data/raw/*.zip` 中包含所需字段的明细 CSV，不再依赖
`C:/Users/fire/all2`。为保持历史分析口径，文件合并后不做跨文件去重，
互动量仍定义为“点赞 + 收藏 + 4 × 评论”。

“疑似买粉”筛选只是历史分析沿用的启发式规则（粉丝量前 10% 且互动率低于 1%），
并非对账号真实性的判定。下列结果均为描述性关联，不应解释为发布时间或发布地的
因果效应。
"""
    ),
    code(
        r'''
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "data" / "raw").exists()
)
DATA_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "chapter06"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def read_repo_csvs(usecols: list[str]) -> pd.DataFrame:
    """Read matching detail CSV members directly from repository ZIP archives."""
    frames = []
    audit = []
    required = set(usecols)
    for archive in sorted(DATA_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if not member.filename.lower().endswith(".csv") or member.file_size == 0:
                    continue
                raw = zf.read(member)
                try:
                    header = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", nrows=0)
                except Exception:
                    continue
                if not required.issubset(header.columns):
                    continue
                frame = pd.read_csv(
                    io.BytesIO(raw), encoding="utf-8-sig", usecols=usecols, low_memory=False
                )
                frames.append(frame)
                audit.append({"压缩包": archive.name, "文件": member.filename, "行数": len(frame)})
    if not frames:
        raise RuntimeError("data/raw 中没有找到包含所需字段的 CSV。")
    pd.DataFrame(audit).to_csv(
        OUT / "6_1_data_file_audit.csv", index=False, encoding="utf-8-sig"
    )
    result = pd.concat(frames, ignore_index=True)
    print(f"读取 {len(audit)} 个 CSV 成员，共 {len(result):,} 行；来源：{DATA_DIR}")
    return result
'''
    ),
    md("## 6.1.1 数据读取、清洗与日期变量"),
    code(
        r'''
cols = [
    "相对准确点赞数", "相对准确评论数", "相对准确收藏数", "相对准确粉丝数",
    "推断笔记发布日期", "笔记发布地",
]
df = read_repo_csvs(cols)
raw_rows = len(df)

df["likes"] = pd.to_numeric(df["相对准确点赞数"], errors="coerce")
df["comments"] = pd.to_numeric(df["相对准确评论数"], errors="coerce")
df["favs"] = pd.to_numeric(df["相对准确收藏数"], errors="coerce")
df["fans"] = pd.to_numeric(df["相对准确粉丝数"], errors="coerce")
df = df.dropna(subset=["likes", "comments", "favs"]).copy()
rows_after_interaction_clean = len(df)

df["engagement"] = df["likes"] + df["favs"] + 4 * df["comments"]
df["fans"] = df["fans"].replace(0, np.nan)
df["engagement_rate"] = df["engagement"] / df["fans"]
fan_threshold = df["fans"].quantile(0.9)
before_filter = len(df)
df = df[~((df["fans"] > fan_threshold) & (df["engagement_rate"] < 0.01))].copy()
rows_after_suspicious_fan_filter = len(df)

df["engagement_log"] = np.log1p(df["engagement"])
df["post_date"] = pd.to_datetime(df["推断笔记发布日期"], errors="coerce")
df = df.dropna(subset=["post_date"]).copy()
rows_with_valid_date = len(df)

df["weekday"] = df["post_date"].dt.dayofweek
df["weekday_name"] = df["post_date"].dt.day_name()
df["location"] = df["笔记发布地"].astype(str)
df["location_valid"] = (
    ~df["location"].isin(["-9999", "nan", ""])
) & df["location"].str.strip().ne("")

print(f"原始行数：{raw_rows:,}")
print(f"互动字段清洗后：{rows_after_interaction_clean:,}")
print(
    "启发式粉丝筛选：剔除 "
    f"{before_filter - rows_after_suspicious_fan_filter:,} 条；"
    f"筛选后 {rows_after_suspicious_fan_filter:,} 条"
)
print(f"日期有效的最终样本：{rows_with_valid_date:,}")
'''
    ),
    md("## 6.1.2 日期类型定义"),
    code(
        r'''
holiday_list = [
    # 2023 春节、清明、劳动节、端午、中秋及国庆
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
    "2023-04-05",
    "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
    "2023-06-22", "2023-06-23", "2023-06-24",
    "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
    # 2024 元旦、春节、清明、劳动节、端午、中秋及国庆
    "2023-12-30", "2023-12-31", "2024-01-01",
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
    "2024-04-04", "2024-04-05", "2024-04-06",
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    "2024-06-08", "2024-06-09", "2024-06-10",
    "2024-09-15", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07",
]
workday_shift = [
    "2023-01-28", "2023-01-29", "2023-04-23", "2023-05-06", "2023-06-25",
    "2023-10-07", "2023-10-08", "2024-02-04", "2024-02-18", "2024-04-07",
    "2024-04-28", "2024-05-11", "2024-09-14", "2024-09-29", "2024-10-12",
]
holiday_set = set(pd.to_datetime(holiday_list))
workday_shift_set = set(pd.to_datetime(workday_shift))

def classify_date(day):
    if day in holiday_set:
        return "节假日"
    if day in workday_shift_set:
        return "调休工作日"
    if day.dayofweek >= 5:
        return "普通周末"
    return "普通工作日"

df["date_type"] = df["post_date"].apply(classify_date)
df["is_holiday"] = df["date_type"].eq("节假日")
display(df["date_type"].value_counts())
'''
    ),
    md("## 6.1.3 星期效应：描述统计、ANOVA 与 Tukey HSD"),
    code(
        r'''
weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
weekday_stats = df.groupby("weekday").agg(
    样本量=("engagement", "count"),
    平均点赞=("likes", "mean"),
    中位点赞=("likes", "median"),
    平均评论=("comments", "mean"),
    平均收藏=("favs", "mean"),
    平均总互动=("engagement", "mean"),
    平均log互动=("engagement_log", "mean"),
).reset_index()
weekday_stats["星期"] = weekday_stats["weekday"].apply(lambda value: weekday_cn[value])
weekday_stats = weekday_stats.set_index("星期")
display(weekday_stats)

groups = [df.loc[df["weekday"].eq(i), "engagement_log"].to_numpy() for i in range(7)]
weekday_f, weekday_p = stats.f_oneway(*groups)
grand_mean = df["engagement_log"].mean()
ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
ss_total = ((df["engagement_log"] - grand_mean) ** 2).sum()
weekday_eta_sq = ss_between / ss_total
tukey = pairwise_tukeyhsd(df["engagement_log"], df["weekday"], alpha=0.05)
print(f"星期 ANOVA：F={weekday_f:.4f}，p={weekday_p:.4e}，eta²={weekday_eta_sq:.8f}")
print(tukey)

weekday_stats.to_csv(OUT / "6_1_weekday_stats.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
means = weekday_stats["平均总互动"]
axes[0].bar(weekday_cn, means, color=sns.color_palette("Blues_d", 7))
axes[0].set(title="各星期平均总互动量", ylabel="平均互动量（点赞+收藏+4×评论）")
for i, value in enumerate(means):
    axes[0].text(i, value + max(means) * 0.02, f"{value:.0f}", ha="center", fontsize=9)

box = axes[1].boxplot(groups, tick_labels=weekday_cn, patch_artist=True)
for patch, color_value in zip(box["boxes"], sns.color_palette("Blues_d", 7)):
    patch.set_facecolor(color_value)
axes[1].set(title="各星期互动量分布（log）", ylabel="log(互动量+1)")

x = np.arange(7)
width = 0.25
axes[2].bar(x - width, weekday_stats["平均点赞"], width, label="点赞", color="#FF6B6B")
axes[2].bar(x, weekday_stats["平均收藏"], width, label="收藏", color="#4ECDC4")
axes[2].bar(x + width, weekday_stats["平均评论"], width, label="评论", color="#FFD93D")
axes[2].set(xticks=x, xticklabels=weekday_cn, title="分指标各星期均值对比")
axes[2].legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "6_1_weekday_effect.png", dpi=180)
plt.show()
'''
    ),
    md("## 6.1.4 节假日效应：描述统计与均值检验"),
    code(
        r'''
date_order = ["普通工作日", "普通周末", "调休工作日", "节假日"]
date_type_stats = df.groupby("date_type").agg(
    样本量=("engagement", "count"),
    平均点赞=("likes", "mean"),
    中位点赞=("likes", "median"),
    平均评论=("comments", "mean"),
    平均收藏=("favs", "mean"),
    平均总互动=("engagement", "mean"),
    平均log互动=("engagement_log", "mean"),
).reindex(date_order)
display(date_type_stats)

holiday_vals = df.loc[df["is_holiday"], "engagement_log"]
non_holiday_vals = df.loc[~df["is_holiday"], "engagement_log"]
holiday_t, holiday_p = stats.ttest_ind(holiday_vals, non_holiday_vals, equal_var=False)
pooled_std = np.sqrt((holiday_vals.var() + non_holiday_vals.var()) / 2)
holiday_cohens_d = (holiday_vals.mean() - non_holiday_vals.mean()) / pooled_std
date_groups = [df.loc[df["date_type"].eq(label), "engagement_log"].to_numpy() for label in date_order]
date_f, date_p = stats.f_oneway(*date_groups)
print(f"节假日 Welch t：t={holiday_t:.4f}，p={holiday_p:.4e}，Cohen's d={holiday_cohens_d:.6f}")
print(f"四类日期 ANOVA：F={date_f:.4f}，p={date_p:.4e}")

date_type_stats.to_csv(OUT / "6_1_date_type_stats.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#95A5A6", "#3498DB", "#E67E22", "#E74C3C"]
bars = axes[0].bar(date_order, date_type_stats["平均log互动"], color=colors)
axes[0].set(title="不同日期类型的平均互动量（log）", ylabel="log(互动量+1)")
for bar, value, count in zip(bars, date_type_stats["平均log互动"], date_type_stats["样本量"]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.3f}\n(n={count:.0f})",
        ha="center",
        fontsize=9,
    )
box = axes[1].boxplot(
    [non_holiday_vals, holiday_vals], tick_labels=["非节假日", "节假日"], patch_artist=True
)
box["boxes"][0].set_facecolor("#3498DB")
box["boxes"][1].set_facecolor("#E74C3C")
axes[1].set(title="节假日与非节假日互动量分布", ylabel="log(互动量+1)")
fig.tight_layout()
fig.savefig(OUT / "6_1_holiday_effect.png", dpi=180)
plt.show()
'''
    ),
    md("## 6.1.5 发布地省份分析"),
    code(
        r'''
df_loc = df[df["location_valid"]].copy()

def extract_province(location):
    location = location.strip()
    for municipality in ["北京", "上海", "天津", "重庆"]:
        if location.startswith(municipality):
            return municipality
    return location[:2] if len(location) >= 2 else location

df_loc["province"] = df_loc["location"].apply(extract_province)
valid_provinces = {
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "内蒙",
    "广西", "西藏", "宁夏", "新疆", "香港", "澳门", "台湾",
}
df_loc = df_loc[df_loc["province"].isin(valid_provinces)].copy()
province_stats = df_loc.groupby("province").agg(
    样本量=("engagement", "count"),
    平均总互动=("engagement", "mean"),
    平均log互动=("engagement_log", "mean"),
    平均粉丝=("fans", "mean"),
).sort_values("样本量", ascending=False)
display(province_stats.head(20))
province_stats.to_csv(OUT / "6_1_province_stats.csv", encoding="utf-8-sig")

province_top = province_stats.head(20).sort_values("平均log互动")
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(province_top)))
ax.barh(province_top.index, province_top["平均log互动"], color=colors)
ax.set(title="样本量前20个省级发布地的平均互动量", xlabel="log(互动量+1)")
for i, (_, row) in enumerate(province_top.iterrows()):
    ax.text(row["平均log互动"] + 0.02, i, f"n={row['样本量']:.0f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "6_1_province_analysis.png", dpi=180)
plt.show()
'''
    ),
    md("## 6.1.6 复现摘要"),
    code(
        r'''
metadata = {
    "raw_rows": int(raw_rows),
    "rows_after_interaction_clean": int(rows_after_interaction_clean),
    "rows_removed_by_suspicious_fan_heuristic": int(before_filter - rows_after_suspicious_fan_filter),
    "rows_after_suspicious_fan_filter": int(rows_after_suspicious_fan_filter),
    "rows_with_valid_date": int(rows_with_valid_date),
    "rows_with_valid_location": int(len(df_loc)),
    "fan_90th_percentile": float(fan_threshold),
    "weekday_anova_f": float(weekday_f),
    "weekday_anova_p": float(weekday_p),
    "weekday_eta_squared": float(weekday_eta_sq),
    "holiday_welch_t": float(holiday_t),
    "holiday_welch_p": float(holiday_p),
    "holiday_cohens_d": float(holiday_cohens_d),
    "four_date_types_anova_f": float(date_f),
    "four_date_types_anova_p": float(date_p),
    "deduplication": "none (historical Chapter 6 convention)",
}
with open(OUT / "6_1_reproduction_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, ensure_ascii=False, indent=2)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
'''
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
