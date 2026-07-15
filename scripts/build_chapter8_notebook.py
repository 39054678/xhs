import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# 第8章 多维度相关性分析（8.1–8.3）

本 Notebook 按指导文件完成：

- **8.1 特征间相关性分析**：Pearson、Spearman、热力图、高相关特征对；
- **8.2 特征与目标变量相关性分析**：数值相关性、爆款点二列相关、Welch t 检验、Mann–Whitney U、效应量与星期分组；
- **8.3 非线性关系探索**：分位数组均值与 LOWESS 曲线。

数据限制：CSV 中没有可靠的图片数量、认证状态和发布小时，因此不分析这些变量。分析只读取“笔记详细信息”文件，排除 ID、筛选中间文件和相关矩阵，并按笔记链接去重。"""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import glob, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 180)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='SimHei')

DATA_DIR = Path(r'C:\\Users\\fire\\all2')
OUTPUT_DIR = Path('outputs/chapter08')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
print('输出目录：', OUTPUT_DIR.resolve())"""))

cells.append(nbf.v4.new_markdown_cell("""## 0. 数据读取、去重与特征构造

互动指标沿用前文定义：`点赞 + 收藏 + 4 × 评论`。为了避免极端偏态，相关性分析使用 `log(互动量+1)` 和 `log(粉丝量+1)`。

“爆款”定义为清洗后样本中互动量位于前 10% 的笔记。这是样本内相对定义，不代表平台官方标准。"""))

cells.append(nbf.v4.new_code_cell("""all_files = glob.glob(str(DATA_DIR / '*.csv'))
detail_files = [f for f in all_files if '笔记详细信息' in os.path.basename(f)]

frames = []
for f in detail_files:
    try:
        frames.append(pd.read_csv(f, encoding='utf-8-sig', low_memory=False))
    except Exception as e:
        print('跳过：', os.path.basename(f), e)

df = pd.concat(frames, ignore_index=True)
before = len(df)

# 优先按唯一链接去重；无链接记录用编号、标题、作者联合去重
link = df['具体链接'].astype('string').str.strip()
has_link = link.notna() & link.ne('') & link.ne('nan')
with_link = df.loc[has_link].drop_duplicates('具体链接')
without_link = df.loc[~has_link].drop_duplicates(['编号', '文案（标题）', '博主名称'])
df = pd.concat([with_link, without_link], ignore_index=True)

print(f'读取详情文件：{len(detail_files)} 个')
print(f'去重前：{before:,} 行；去重后：{len(df):,} 行；移除：{before-len(df):,} 行')"""))

cells.append(nbf.v4.new_code_cell("""def numeric(col):
    return pd.to_numeric(df[col], errors='coerce')

df['likes'] = numeric('相对准确点赞数')
df['comments'] = numeric('相对准确评论数')
df['favs'] = numeric('相对准确收藏数')
df['fans'] = numeric('相对准确粉丝数')
df = df.dropna(subset=['likes', 'comments', 'favs']).copy()
df[['likes', 'comments', 'favs']] = df[['likes', 'comments', 'favs']].clip(lower=0)

df['engagement'] = df['likes'] + df['favs'] + 4 * df['comments']
df['engagement_log'] = np.log1p(df['engagement'])
df['log_fans'] = np.log1p(df['fans'].clip(lower=0))

title = df['文案（标题）'].fillna('').astype(str)
body = df['文案（正文）'].fillna('').astype(str)
text = (title + ' ' + body).str.strip()
df['title_len'] = title.str.len()
df['body_len'] = body.str.len()
df['content_len'] = text.str.len()
df['tag_count'] = df['标签词'].fillna('').astype(str).apply(lambda s: 0 if not s.strip() else len([x for x in s.split(',') if x.strip()]))
df['exclamation_count'] = text.str.count(r'[!！]')
df['question_count'] = text.str.count(r'[?？]')
df['emoji_count'] = text.apply(lambda s: len(re.findall(r'[\\U0001F300-\\U0001FAFF]', s)))
df['emotion_positive'] = pd.to_numeric(df['情感预测'], errors='coerce').clip(0, 1)
df['emotion_extremity'] = (df['emotion_positive'] - 0.5).abs()

post_date = pd.to_datetime(df['推断笔记发布日期'], errors='coerce')
df['weekday'] = post_date.dt.dayofweek
location = df['笔记发布地'].fillna('').astype(str).str.strip()
df['has_location'] = (~location.isin(['', 'nan', '-9999'])).astype(int)
df['category'] = np.where(df['具体链接'].notna(), '笔记', '未知')  # 文件合并后无可靠来源列，不用于类别推断

# 标题-正文词汇重合度（简单且可解释，不等同于语义相似度）
token_pattern = re.compile(r'[\\u4e00-\\u9fffA-Za-z0-9]+')
def title_body_overlap(t, b):
    a = set(token_pattern.findall(t.lower()))
    c = set(token_pattern.findall(b.lower()))
    return len(a & c) / len(a) if a else 0.0
df['title_body_overlap'] = [title_body_overlap(t, b) for t, b in zip(title, body)]

viral_cutoff = df['engagement'].quantile(0.90)
df['is_viral'] = (df['engagement'] >= viral_cutoff).astype(int)
print(f'最终样本：{len(df):,}；爆款阈值：互动量 ≥ {viral_cutoff:.1f}；爆款比例：{df.is_viral.mean():.2%}')"""))

cells.append(nbf.v4.new_markdown_cell("""## 8.1 特征间相关性分析

同时报告 Pearson（线性关系）和 Spearman（单调关系）。绝对 Pearson 相关系数大于 0.8 的特征对单独列出，供后续共线性处理参考。"""))

cells.append(nbf.v4.new_code_cell("""features = [
    'log_fans', 'title_len', 'body_len', 'content_len', 'tag_count',
    'exclamation_count', 'question_count', 'emoji_count',
    'emotion_positive', 'emotion_extremity', 'title_body_overlap',
    'weekday', 'has_location'
]

pearson = df[features].corr(method='pearson')
spearman = df[features].corr(method='spearman')
pearson.to_csv(OUTPUT_DIR / '8_1_pearson_features.csv', encoding='utf-8-sig')
spearman.to_csv(OUTPUT_DIR / '8_1_spearman_features.csv', encoding='utf-8-sig')

mask = np.triu(np.ones_like(pearson, dtype=bool))
plt.figure(figsize=(13, 10))
sns.heatmap(pearson, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True, linewidths=.4)
plt.title('8.1 数值特征 Pearson 相关系数')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8_1_pearson_heatmap.png', dpi=180)
plt.show()

pairs = []
for i, a in enumerate(features):
    for b in features[i+1:]:
        r = pearson.loc[a, b]
        if pd.notna(r) and abs(r) >= 0.8:
            pairs.append((a, b, r))
high_corr = pd.DataFrame(pairs, columns=['feature_1', 'feature_2', 'pearson_r']).sort_values('pearson_r', key=abs, ascending=False) if pairs else pd.DataFrame(columns=['feature_1','feature_2','pearson_r'])
high_corr.to_csv(OUTPUT_DIR / '8_1_high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
display(high_corr if len(high_corr) else pd.DataFrame({'结论':['未发现 |r| ≥ 0.8 的特征对']}))"""))

cells.append(nbf.v4.new_markdown_cell("""## 8.2 特征与目标变量相关性分析

数值目标为 `engagement_log`。二值目标为样本内互动量前 10% 的“爆款”。

注意：大样本下很小的差异也可能显著，因此同时报告相关系数、Cohen's d 和 FDR 校正后的 p 值，不只看显著性。"""))

cells.append(nbf.v4.new_code_cell("""rows = []
for x in features:
    z = df[[x, 'engagement_log']].dropna()
    pr, pp = stats.pearsonr(z[x], z['engagement_log']) if z[x].nunique() > 1 else (np.nan, np.nan)
    sr, sp = stats.spearmanr(z[x], z['engagement_log']) if z[x].nunique() > 1 else (np.nan, np.nan)
    rows.append([x, len(z), pr, pp, sr, sp])

target_corr = pd.DataFrame(rows, columns=['feature','n','pearson_r','pearson_p','spearman_rho','spearman_p'])
target_corr['pearson_p_fdr'] = multipletests(target_corr['pearson_p'].fillna(1), method='fdr_bh')[1]
target_corr['spearman_p_fdr'] = multipletests(target_corr['spearman_p'].fillna(1), method='fdr_bh')[1]
target_corr = target_corr.sort_values('spearman_rho', key=abs, ascending=False)
target_corr.to_csv(OUTPUT_DIR / '8_2_numeric_target_correlations.csv', index=False, encoding='utf-8-sig')
display(target_corr.round(5))

plt.figure(figsize=(9, 6))
plot_corr = target_corr.sort_values('spearman_rho')
plt.barh(plot_corr['feature'], plot_corr['spearman_rho'], color=np.where(plot_corr['spearman_rho'] >= 0, '#D95F5F', '#4C78A8'))
plt.axvline(0, color='black', lw=.8)
plt.xlabel('Spearman ρ with log engagement')
plt.title('8.2 数值特征与互动量的相关性')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8_2_target_correlation_ranking.png', dpi=180)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean()-b.mean()) / pooled if pooled > 0 else np.nan

rows = []
for x in features:
    z = df[[x, 'is_viral']].dropna()
    viral = z.loc[z.is_viral == 1, x]
    normal = z.loc[z.is_viral == 0, x]
    if z[x].nunique() <= 1:
        continue
    r_pb, p_pb = stats.pointbiserialr(z['is_viral'], z[x])
    t, p_t = stats.ttest_ind(viral, normal, equal_var=False)
    u, p_u = stats.mannwhitneyu(viral, normal, alternative='two-sided')
    rows.append([x, len(z), normal.mean(), viral.mean(), r_pb, p_pb, t, p_t, u, p_u, cohens_d(viral, normal)])

viral_tests = pd.DataFrame(rows, columns=[
    'feature','n','nonviral_mean','viral_mean','point_biserial_r','point_biserial_p',
    'welch_t','welch_p','mannwhitney_u','mannwhitney_p','cohens_d_viral_minus_nonviral'
])
for c in ['point_biserial_p','welch_p','mannwhitney_p']:
    viral_tests[c + '_fdr'] = multipletests(viral_tests[c].fillna(1), method='fdr_bh')[1]
viral_tests = viral_tests.sort_values('cohens_d_viral_minus_nonviral', key=abs, ascending=False)
viral_tests.to_csv(OUTPUT_DIR / '8_2_viral_group_tests.csv', index=False, encoding='utf-8-sig')
display(viral_tests.round(5))"""))

cells.append(nbf.v4.new_code_cell("""weekday_names = ['周一','周二','周三','周四','周五','周六','周日']
weekday_summary = df.dropna(subset=['weekday']).groupby('weekday').agg(
    样本量=('engagement_log','size'),
    平均log互动=('engagement_log','mean'),
    中位log互动=('engagement_log','median'),
    爆款比例=('is_viral','mean')
).reindex(range(7))
weekday_summary.index = weekday_names
weekday_summary.to_csv(OUTPUT_DIR / '8_2_weekday_summary.csv', encoding='utf-8-sig')
display(weekday_summary.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df.dropna(subset=['weekday']), x='weekday', y='engagement_log', showfliers=False, ax=axes[0])
axes[0].set_xticklabels(weekday_names)
axes[0].set_title('星期与 log 互动量')
axes[0].set_xlabel('')
axes[1].bar(weekday_names, weekday_summary['爆款比例'], color='#4C78A8')
axes[1].set_title('各星期爆款比例')
axes[1].set_ylabel('爆款比例')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8_2_weekday_comparison.png', dpi=180)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## 8.3 非线性关系探索

对关键连续特征绘制分位数组均值和 LOWESS。散点最多随机抽取 20,000 条以控制运行时间，但分位数组均值使用全部有效数据。LOWESS 用于探索形状，不能单独证明因果关系。"""))

cells.append(nbf.v4.new_code_cell("""nonlinear_features = ['log_fans', 'content_len', 'tag_count', 'emotion_positive', 'emotion_extremity', 'title_body_overlap']

def nonlinear_plot(data, x, bins=20, sample_n=20000):
    z = data[[x, 'engagement_log']].replace([np.inf, -np.inf], np.nan).dropna()
    z = z[z[x].between(z[x].quantile(.005), z[x].quantile(.995))]
    sample = z.sample(min(sample_n, len(z)), random_state=RANDOM_STATE).sort_values(x)
    smooth = lowess(sample['engagement_log'], sample[x], frac=.25, it=1, return_sorted=True)
    try:
        z['bin'] = pd.qcut(z[x], bins, duplicates='drop')
        grouped = z.groupby('bin', observed=True).agg(x_mean=(x,'mean'), y_mean=('engagement_log','mean'), n=('engagement_log','size')).reset_index(drop=True)
    except ValueError:
        grouped = pd.DataFrame(columns=['x_mean','y_mean','n'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample[x], sample['engagement_log'], s=5, alpha=.05, color='gray', rasterized=True)
    ax.plot(smooth[:,0], smooth[:,1], color='#D62728', lw=2.5, label='LOWESS')
    if len(grouped):
        ax.plot(grouped.x_mean, grouped.y_mean, 'o-', color='#1F77B4', ms=4, label='分位数组均值')
    ax.set_xlabel(x); ax.set_ylabel('log(互动量+1)'); ax.set_title(f'8.3 {x} 与互动量的非线性关系')
    ax.legend(); plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'8_3_lowess_{x}.png', dpi=180)
    plt.show()
    grouped.to_csv(OUTPUT_DIR / f'8_3_bins_{x}.csv', index=False, encoding='utf-8-sig')
    return grouped

bin_results = {x: nonlinear_plot(df, x) for x in nonlinear_features}"""))

cells.append(nbf.v4.new_markdown_cell("""## 自动摘要与解释规则

- `|r| < 0.10`：很弱；`0.10–0.30`：弱；`0.30–0.50`：中等；`≥0.50`：较强（仅为便于描述的经验阈值）。
- 大样本时不能只依据 p 值，应结合相关系数、Cohen's d、图形趋势和业务意义。
- 相关关系不代表因果。粉丝量、账号差异、发布时间及内容类别可能产生混杂。
- 如果 LOWESS 明显弯曲，可在后续模型中加入平方项、样条或树模型进一步检验。"""))

cells.append(nbf.v4.new_code_cell("""def strength(v):
    a = abs(v)
    return '很弱' if a < .10 else ('弱' if a < .30 else ('中等' if a < .50 else '较强'))

top = target_corr.dropna(subset=['spearman_rho']).head(8).copy()
print('与 log 互动量 Spearman 相关性绝对值最高的特征：')
for _, r in top.iterrows():
    direction = '正' if r.spearman_rho > 0 else '负'
    print(f"- {r.feature}: ρ={r.spearman_rho:.3f}（{strength(r.spearman_rho)}{direction}相关，FDR p={r.spearman_p_fdr:.3g}）")

print('\\n文件已输出到：', OUTPUT_DIR.resolve())
print('\\n注意：以上结果是探索性相关分析，不能表述为某特征“导致”互动量变化。')"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3'}
}
NOTEBOOK_PATH = Path('notebooks/chapter08/chapter8_correlation_analysis.ipynb')
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f'Created {NOTEBOOK_PATH}')
