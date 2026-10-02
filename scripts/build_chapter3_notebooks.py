from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = ROOT / "notebooks" / "chapter03"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


COMMON_SETUP = r'''
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "data" / "raw").exists()
)
DATA_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "chapter03"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
logging.getLogger("jieba").setLevel(logging.CRITICAL)


def read_repo_csvs(usecols: list[str], audit_name: str) -> pd.DataFrame:
    """Read the 76 historical CSV members directly from the three repository ZIPs."""
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
    pd.DataFrame(audit).to_csv(OUT / audit_name, index=False, encoding="utf-8-sig")
    result = pd.concat(frames, ignore_index=True)
    print(f"读取 {len(audit)} 个 CSV 成员，共 {len(result):,} 行；来源：{DATA_DIR}")
    return result
'''


chapter31_cells = [
    md(
        """
# 3.1 文本内容深度、可读性与情感

本 Notebook 是旧版 `code3.1.6.ipynb` 的可复现整理版。它直接读取仓库
`data/raw/*.zip`，不再依赖 `C:/Users/fire/all2`。为保持与历史报告口径一致，
本节仍合并 ZIP 中 76 份符合字段要求的 CSV，不做跨文件去重，并固定随机种子 42。

历史 Notebook 依赖未在项目中声明的 `cntext`。当前版本把研究中实际使用的
Gunning Fog 指标按 `cntext` 官方实现的分句、分词和计数规则写在 Notebook 内，
从而不需要外部数据或在线服务。该指标是描述性代理，不应解释为内容质量或因果效应。
"""
    ),
    code(COMMON_SETUP),
    md("## 3.1.1 数据读取与基础清洗"),
    code(
        r'''
cols = [
    "文案（标题）", "文案（正文）", "文本", "情感预测",
    "相对准确点赞数", "相对准确评论数", "相对准确收藏数", "相对准确粉丝数",
]
df = read_repo_csvs(cols, "3_1_data_file_audit.csv")
raw_rows = len(df)

df["文本"] = df["文本"].astype(str)
df["likes"] = pd.to_numeric(df["相对准确点赞数"], errors="coerce")
df["comments"] = pd.to_numeric(df["相对准确评论数"], errors="coerce")
df["favs"] = pd.to_numeric(df["相对准确收藏数"], errors="coerce")
df["fans"] = pd.to_numeric(df["相对准确粉丝数"], errors="coerce")
df = df.dropna(subset=["likes", "comments", "favs"]).copy()
rows_after_interaction_clean = len(df)

df["engagement"] = df["likes"] + df["favs"] + 4 * df["comments"]
df["engagement_log"] = np.log1p(df["engagement"])
df["fans"] = df["fans"].replace(0, np.nan)
df["engagement_rate"] = df["engagement"] / df["fans"]

fan_threshold = df["fans"].quantile(0.9)
df = df[~((df["fans"] > fan_threshold) & (df["engagement_rate"] < 0.01))].copy()
rows_after_suspicious_fan_filter = len(df)

df["title"] = df["文案（标题）"].astype(str)
df["body"] = df["文案（正文）"].astype(str)
df["title_len"] = df["title"].str.len()
df["body_len"] = df["body"].str.len()
df["content_len"] = df["文本"].str.len()
df["emotion_strength"] = pd.to_numeric(df["情感预测"], errors="coerce")
df["emotion_extreme"] = (df["emotion_strength"] - 0.5).abs()

print(f"原始行数：{raw_rows:,}")
print(f"互动字段清洗后：{rows_after_interaction_clean:,}")
print(f"疑似买粉规则过滤后：{rows_after_suspicious_fan_filter:,}")
print(f"粉丝量第 90 百分位：{fan_threshold:,.1f}")
'''
    ),
    md("## 3.1.2 文本长度、情感与互动分布"),
    code(
        r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].hexbin(df["content_len"], df["engagement_log"], gridsize=50, mincnt=1, cmap="Blues")
axes[0].set(title="文本长度与互动", xlabel="文本长度", ylabel="log(互动量+1)")
axes[1].hexbin(df["emotion_strength"], df["engagement_log"], gridsize=50, mincnt=1, cmap="Oranges")
axes[1].set(title="情感得分与互动", xlabel="情感得分", ylabel="log(互动量+1)")
axes[2].hist(df["engagement_log"], bins=100, color="#4C78A8")
axes[2].set(title="对数互动量分布", xlabel="log(互动量+1)", ylabel="样本数")
fig.tight_layout()
fig.savefig(OUT / "3_1_text_emotion_engagement.png", dpi=180)
plt.show()

vis = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(
    data=vis, x="content_len", y="engagement_log", lowess=True, ax=ax,
    scatter_kws={"s": 5, "alpha": 0.08}, line_kws={"color": "#B6574B", "lw": 2},
)
ax.set(title="文本长度与互动的 LOWESS 趋势", xlabel="文本长度", ylabel="log(互动量+1)")
fig.tight_layout()
fig.savefig(OUT / "3_1_content_length_lowess.png", dpi=180)
plt.show()
'''
    ),
    md(
        """
## 3.1.3 `cntext` 兼容的可读性指标

下列函数复现历史代码所调用的 `cntext.readability(..., lang="chinese")`
中的 `fog_index` 行为，包括其中文分句、jieba 分词和复杂词口径。官方源码：
https://github.com/hidadeng/cntext/blob/main/cntext/stats/readability.py

这里固定计算同一批 50,000 条随机样本；空白文本记为缺失。由于这是特定软件实现的
复杂度代理，结果只用于复现历史模型，不作为通用中文可读性标准。
"""
    ),
    code(
        r'''
def cn_seg_sent(text: str) -> list[str]:
    text = re.sub(r"([。！；？;\?])([^”’])", "[[end]]", text)
    text = re.sub(r"([。！？\?][”’])([^，。！？\?])", "[[end]]", text)
    text = re.sub(r"\s", "", text)
    return text.split("[[end]]")


def cntext_compatible_fog_index(text: str, syllables: int = 3) -> float:
    if not isinstance(text, str) or not text.strip():
        return np.nan
    sentences = cn_seg_sent(text)
    total_sentences = len(sentences) or 1
    total_words = 0
    total_complex_words = 0
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        for word in words:
            if len(word) >= syllables:
                total_complex_words += 1
    # 与历史 cntext 实现保持一致：中文分支未累加 total_words，随后以 1 防止除零。
    total_words = total_words or 1
    return round(
        0.4 * (total_words / total_sentences)
        + 0.4 * (100 * total_complex_words / total_words),
        2,
    )


read_sample_n = min(50000, len(df))
read_idx = df.sample(read_sample_n, random_state=RANDOM_STATE).index
df.loc[read_idx, "readability"] = df.loc[read_idx, "文本"].apply(cntext_compatible_fog_index)
print(f"可读性抽样：{read_sample_n:,}；有效值：{df['readability'].notna().sum():,}")
'''
    ),
    md("## 3.1.4 OLS 与去除互动量最高 1% 的稳健性检验"),
    code(
        r'''
model_cols = [
    "engagement_log", "content_len", "emotion_strength", "emotion_extreme", "readability"
]
df_model = df.dropna(subset=model_cols).copy()
df_model["content_len_sq"] = df_model["content_len"] ** 2
model_sample = df_model.sample(min(20000, len(df_model)), random_state=RANDOM_STATE)
X = sm.add_constant(model_sample[[
    "content_len", "content_len_sq", "emotion_strength", "emotion_extreme", "readability"
]])
model = sm.OLS(model_sample["engagement_log"], X).fit()
print(f"可建模样本：{len(df_model):,}；回归抽样：{len(model_sample):,}")
print(model.summary())

robust = df[df["engagement"] < df["engagement"].quantile(0.99)].copy()
robust = robust.dropna(subset=model_cols)
robust["content_len_sq"] = robust["content_len"] ** 2
robust_sample = robust.sample(min(20000, len(robust)), random_state=RANDOM_STATE)
Xr = sm.add_constant(robust_sample[[
    "content_len", "content_len_sq", "emotion_strength", "emotion_extreme", "readability"
]])
model_r = sm.OLS(robust_sample["engagement_log"], Xr).fit()
print(f"\n稳健性可建模样本：{len(robust):,}；回归抽样：{len(robust_sample):,}")
print(model_r.summary())

def coefficient_table(fitted, label):
    ci = fitted.conf_int()
    return pd.DataFrame({
        "模型": label,
        "变量": fitted.params.index,
        "系数": fitted.params.values,
        "标准误": fitted.bse.values,
        "t值": fitted.tvalues.values,
        "p值": fitted.pvalues.values,
        "置信区间下限": ci[0].values,
        "置信区间上限": ci[1].values,
    })

coef = pd.concat([
    coefficient_table(model, "主模型"), coefficient_table(model_r, "去除互动Top1%")
], ignore_index=True)
coef.to_csv(OUT / "3_1_ols_coefficients.csv", index=False, encoding="utf-8-sig")
'''
    ),
    md("## 3.1.5 分组结果与复现摘要"),
    code(
        r'''
df["len_group"] = pd.qcut(df["content_len"], 10, duplicates="drop")
length_groups = df.groupby("len_group", observed=True).agg(
    样本量=("engagement_log", "size"),
    平均文本长度=("content_len", "mean"),
    平均log互动=("engagement_log", "mean"),
    平均互动=("engagement", "mean"),
).reset_index()
length_groups["长度区间"] = length_groups["len_group"].astype(str)
length_groups.drop(columns="len_group").to_csv(
    OUT / "3_1_length_groups.csv", index=False, encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(length_groups["平均文本长度"], length_groups["平均log互动"], marker="o")
ax.set(title="文本长度分位组与平均互动", xlabel="组内平均文本长度", ylabel="平均 log(互动量+1)")
fig.tight_layout()
fig.savefig(OUT / "3_1_length_groups.png", dpi=180)
plt.show()

emotion = df.dropna(subset=["emotion_strength", "emotion_extreme"]).copy()
emotion["emotion_group"] = pd.qcut(emotion["emotion_strength"], 5, duplicates="drop")
emotion["extreme_group"] = pd.qcut(emotion["emotion_extreme"], 5, duplicates="drop")
emotion_summary = pd.concat([
    emotion.groupby("emotion_group", observed=True)["engagement"].agg(["size", "mean"])
        .rename_axis("分组").reset_index().assign(指标="情感得分"),
    emotion.groupby("extreme_group", observed=True)["engagement"].agg(["size", "mean"])
        .rename_axis("分组").reset_index().assign(指标="情绪极端程度"),
], ignore_index=True)
emotion_summary["分组"] = emotion_summary["分组"].astype(str)
emotion_summary.to_csv(OUT / "3_1_emotion_groups.csv", index=False, encoding="utf-8-sig")

metadata = {
    "raw_rows": int(raw_rows),
    "rows_after_interaction_clean": int(rows_after_interaction_clean),
    "rows_after_suspicious_fan_filter": int(rows_after_suspicious_fan_filter),
    "readability_sample_rows": int(read_sample_n),
    "readability_valid_rows": int(df["readability"].notna().sum()),
    "ols_rows": int(len(model_sample)),
    "ols_r_squared": float(model.rsquared),
    "robust_ols_rows": int(len(robust_sample)),
    "robust_ols_r_squared": float(model_r.rsquared),
    "readability_method": "cntext-compatible fog_index behavior, vendored from official implementation",
    "deduplication": "none (historical Chapter 3.1 convention)",
}
with open(OUT / "3_1_reproduction_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
display(emotion_summary)
'''
    ),
]


chapter32_cells = [
    md(
        """
# 3.2 主题相关性与信息价值分析

本 Notebook 是旧版 `3.2.1.ipynb` 的可复现整理版。它直接读取仓库中的三个原始
ZIP，按历史口径合并其中 76 份明细 CSV，并固定抽样随机种子。信息密度基于
TF-IDF 高权重词占有效词数的比例；标题匹配度是标题词集合与正文词集合的重合比例。

这些指标衡量词汇分布而非语义一致性，回归关系也不构成因果证据。
"""
    ),
    code(COMMON_SETUP + "\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.preprocessing import StandardScaler\nfrom statsmodels.stats.outliers_influence import variance_inflation_factor"),
    md("## 3.2.1 数据读取、清洗与固定抽样"),
    code(
        r'''
cols = [
    "文案（标题）", "文案（正文）", "情感预测",
    "相对准确点赞数", "相对准确评论数", "相对准确收藏数", "相对准确粉丝数",
]
df = read_repo_csvs(cols, "3_2_data_file_audit.csv")
raw_rows = len(df)
df = df.dropna(subset=["文案（正文）"]).copy()
rows_with_body = len(df)

df["文案（标题）"] = df["文案（标题）"].astype(str).replace("nan", "")
df["文案（正文）"] = df["文案（正文）"].astype(str).replace("nan", "")
df["likes"] = pd.to_numeric(df["相对准确点赞数"], errors="coerce")
df["comments"] = pd.to_numeric(df["相对准确评论数"], errors="coerce")
df["favs"] = pd.to_numeric(df["相对准确收藏数"], errors="coerce")
df["fans"] = pd.to_numeric(df["相对准确粉丝数"], errors="coerce")
df = df.dropna(subset=["likes", "comments", "favs"]).copy()

df["engagement"] = df["likes"] + df["favs"] + 4 * df["comments"]
df["engagement_log"] = np.log1p(df["engagement"])
df["fans"] = df["fans"].replace(0, np.nan)
df["engagement_rate"] = df["engagement"] / df["fans"]
fan_threshold = df["fans"].quantile(0.9)
df = df[~((df["fans"] > fan_threshold) & (df["engagement_rate"] < 0.01))].copy()
rows_after_filter = len(df)

df["content_len"] = df["文案（正文）"].str.len()
df["emotion_strength"] = pd.to_numeric(df["情感预测"], errors="coerce").fillna(0.5)
df["emotion_extreme"] = (df["emotion_strength"] - 0.5).abs()

max_sample = 25000
df_work = df.sample(min(max_sample, len(df)), random_state=RANDOM_STATE).copy()
print(f"原始行数：{raw_rows:,}；正文非空：{rows_with_body:,}；过滤后：{rows_after_filter:,}")
print(f"信息价值分析样本：{len(df_work):,}")
'''
    ),
    md("## 3.2.2 分词、信息密度与标题匹配度"),
    code(
        r'''
STOPWORDS = {"的", "了", "和", "是", "就", "都", "而", "及", "与", "这", "那", "被", "在", "给"}

def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in jieba.lcut(text) if w not in STOPWORDS and len(w.strip()) > 1]

df_work["title_tokens"] = df_work["文案（标题）"].apply(tokenize)
df_work["content_tokens"] = df_work["文案（正文）"].apply(tokenize)
df_work["content_joined"] = df_work["content_tokens"].apply(" ".join)

def title_match(row):
    title_set = set(row["title_tokens"])
    return len(title_set & set(row["content_tokens"])) / len(title_set) if title_set else 0.0

df_work["title_match"] = df_work.apply(title_match, axis=1)

vectorizer = TfidfVectorizer(max_features=10000, min_df=5)
tfidf_matrix = vectorizer.fit_transform(df_work["content_joined"])
info_density = []
for i in range(tfidf_matrix.shape[0]):
    weights = tfidf_matrix.getrow(i).data
    total_words = len(df_work["content_tokens"].iloc[i])
    info_density.append(float(np.sum(weights > 0.1) / total_words) if total_words else 0.0)
df_work["info_density"] = info_density

print(f"TF-IDF 词表：{len(vectorizer.vocabulary_):,}")
display(df_work[["info_density", "title_match"]].describe().T)
'''
    ),
    md("## 3.2.3 二维密度图"),
    code(
        r'''
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
hb1 = axes[0].hexbin(df_work["info_density"], df_work["engagement_log"], gridsize=50, cmap="Blues", mincnt=1)
axes[0].set(title="信息密度与互动表现", xlabel="信息密度", ylabel="log(互动量+1)")
fig.colorbar(hb1, ax=axes[0], label="样本数量")
hb2 = axes[1].hexbin(df_work["title_match"], df_work["engagement_log"], gridsize=50, cmap="Oranges", mincnt=1)
axes[1].set(title="标题匹配度与互动表现", xlabel="标题匹配度", ylabel="log(互动量+1)")
fig.colorbar(hb2, ax=axes[1], label="样本数量")
fig.tight_layout()
fig.savefig(OUT / "3_2_info_value_hexbin.png", dpi=180)
plt.show()
'''
    ),
    md("## 3.2.4 标准化 OLS 与 VIF"),
    code(
        r'''
reg = df_work.sample(min(20000, len(df_work)), random_state=RANDOM_STATE).copy()
continuous = ["content_len", "emotion_strength", "emotion_extreme", "info_density", "title_match"]
reg[continuous] = StandardScaler().fit_transform(reg[continuous])
reg["content_len_sq"] = reg["content_len"] ** 2

formula = (
    "engagement_log ~ content_len + content_len_sq + emotion_strength + "
    "emotion_extreme + info_density + title_match"
)
model = __import__("statsmodels.formula.api", fromlist=["ols"]).ols(formula, data=reg).fit()
print(model.summary())

vif_X = sm.add_constant(reg[[
    "content_len", "content_len_sq", "emotion_strength", "emotion_extreme", "info_density", "title_match"
]])
vif = pd.DataFrame({
    "变量": vif_X.columns,
    "VIF": [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])],
})
display(vif)

ci = model.conf_int()
coef = pd.DataFrame({
    "变量": model.params.index,
    "系数": model.params.values,
    "标准误": model.bse.values,
    "t值": model.tvalues.values,
    "p值": model.pvalues.values,
    "置信区间下限": ci[0].values,
    "置信区间上限": ci[1].values,
})
coef.to_csv(OUT / "3_2_ols_coefficients.csv", index=False, encoding="utf-8-sig")
vif.to_csv(OUT / "3_2_vif.csv", index=False, encoding="utf-8-sig")

metadata = {
    "raw_rows": int(raw_rows),
    "rows_with_nonmissing_body": int(rows_with_body),
    "rows_after_suspicious_fan_filter": int(rows_after_filter),
    "feature_sample_rows": int(len(df_work)),
    "ols_rows": int(len(reg)),
    "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
    "ols_r_squared": float(model.rsquared),
    "ols_adjusted_r_squared": float(model.rsquared_adj),
    "condition_number": float(model.condition_number),
    "deduplication": "none (historical Chapter 3.2 convention)",
}
with open(OUT / "3_2_reproduction_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
'''
    ),
]


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )


CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
nbf.write(notebook(chapter31_cells), CHAPTER_DIR / "code3.1.6.ipynb")
nbf.write(notebook(chapter32_cells), CHAPTER_DIR / "3.2.1.ipynb")
print(CHAPTER_DIR / "code3.1.6.ipynb")
print(CHAPTER_DIR / "3.2.1.ipynb")
