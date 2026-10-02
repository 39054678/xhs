"""Run out-of-time and author-group robustness checks for the heat models.

The script reads the repository's raw ZIP archives, applies the same target and
feature definitions as chapters 10 and 11, and writes machine-readable results
to ``outputs/chapter12``.  It deliberately keeps the original random-split
experiments untouched.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "chapter12"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def read_detail_csvs() -> pd.DataFrame:
    required = {
        "文案（标题）",
        "文案（正文）",
        "博主名称",
        "相对准确点赞数",
        "相对准确评论数",
        "相对准确收藏数",
    }
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for archive in sorted(DATA_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if not member.filename.lower().endswith(".csv") or member.file_size == 0:
                    continue
                try:
                    frame = pd.read_csv(
                        io.BytesIO(zf.read(member)), encoding="utf-8-sig", low_memory=False
                    )
                except Exception:
                    continue
                if not required.issubset(frame.columns):
                    continue
                frame["数据来源"] = archive.stem
                frames.append(frame)
                audit.append(
                    {"压缩包": archive.name, "文件": member.filename, "行数": len(frame)}
                )
    if not frames:
        raise RuntimeError("没有在 data/raw ZIP 中找到包含所需字段的详情 CSV。")
    pd.DataFrame(audit).to_csv(
        OUT / "12_1_data_file_audit.csv", index=False, encoding="utf-8-sig"
    )
    return pd.concat(frames, ignore_index=True)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    raw_rows = len(frame)
    if "具体链接" in frame:
        link = frame["具体链接"].astype("string").str.strip()
        has_link = link.notna() & link.ne("") & link.ne("nan")
        linked = frame.loc[has_link].drop_duplicates("具体链接")
        fallback = [c for c in ["编号", "文案（标题）", "博主名称"] if c in frame]
        unlinked = frame.loc[~has_link].drop_duplicates(fallback)
        frame = pd.concat([linked, unlinked], ignore_index=True)

    likes = numeric(frame, "相对准确点赞数")
    comments = numeric(frame, "相对准确评论数")
    favourites = numeric(frame, "相对准确收藏数")
    fans_column = "相对准确粉丝数" if "相对准确粉丝数" in frame else "粉丝量"
    fans = numeric(frame, fans_column)
    valid = (
        likes.notna()
        & comments.notna()
        & favourites.notna()
        & likes.ge(0)
        & comments.ge(0)
        & favourites.ge(0)
    )
    frame = frame.loc[valid].copy()
    likes = likes.loc[valid]
    comments = comments.loc[valid]
    favourites = favourites.loc[valid]
    fans = fans.loc[valid]

    heat = (likes + favourites + 4 * comments).reset_index(drop=True)
    title = frame["文案（标题）"].fillna("").astype(str)
    body = frame["文案（正文）"].fillna("").astype(str)
    tags = frame.get("标签词", pd.Series("", index=frame.index)).fillna("").astype(str)
    full_text = title + " " + body
    date_source = frame.get("推断笔记发布日期", frame.get("笔记发布日期"))
    post_date = pd.to_datetime(date_source, errors="coerce").reset_index(drop=True)
    emotion = (
        numeric(frame, "情感预测")
        if "情感预测" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    location = (
        frame.get("笔记发布地", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    features = pd.DataFrame(
        {
            "log_fans": np.log1p(fans.clip(lower=0)).to_numpy(),
            "title_len": title.str.len().to_numpy(),
            "body_len": body.str.len().to_numpy(),
            "tag_count": tags.apply(
                lambda value: len([x for x in re.split(r"[,，#\s]+", value) if x])
            ).to_numpy(),
            "exclamation_count": full_text.str.count(r"[!！]").to_numpy(),
            "question_count": full_text.str.count(r"[?？]").to_numpy(),
            "emoji_count": full_text.apply(
                lambda value: len(re.findall(r"[\U0001F300-\U0001FAFF]", value))
            ).to_numpy(),
            "emotion_positive": emotion.to_numpy(),
            "weekday": post_date.dt.dayofweek.to_numpy(),
            "has_location": (~location.isin(["", "nan", "-9999"])).astype(int).to_numpy(),
        }
    )
    authors = frame["博主名称"].fillna("").astype(str).str.strip().reset_index(drop=True)
    cutoff = float(heat.quantile(0.80))
    high_heat = heat.ge(cutoff).astype(int)
    metadata = {
        "raw_rows": int(raw_rows),
        "deduplicated_usable_rows": int(len(frame)),
        "valid_date_rows": int(post_date.notna().sum()),
        "valid_author_rows": int(authors.ne("").sum()),
        "unique_authors": int(authors.loc[authors.ne("")].nunique()),
        "viral_cutoff": cutoff,
        "features": list(features.columns),
        "target_definition": "点赞数 + 收藏数 + 4×评论数",
    }
    return features, heat, high_heat, post_date, authors, metadata


def temporal_split(post_date: pd.Series) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_index = np.flatnonzero(post_date.notna().to_numpy())
    ordered = valid_index[np.argsort(post_date.iloc[valid_index].to_numpy())]
    boundary = int(len(ordered) * 0.80)
    # Keep every record from the boundary date in the test set so that a
    # calendar day never appears on both sides of the temporal split.
    test_start = post_date.iloc[ordered[boundary]].normalize()
    train_index = valid_index[(post_date.iloc[valid_index] < test_start).to_numpy()]
    test_index = valid_index[(post_date.iloc[valid_index] >= test_start).to_numpy()]
    details = {
        "train_end": str(post_date.iloc[train_index].max().date()),
        "test_start": str(post_date.iloc[test_index].min().date()),
        "test_end": str(post_date.iloc[test_index].max().date()),
    }
    return train_index, test_index, details


def author_group_split(authors: pd.Series) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_index = np.flatnonzero(authors.ne("").to_numpy())
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    local_train, local_test = next(
        splitter.split(valid_index, groups=authors.iloc[valid_index].to_numpy())
    )
    train_index = valid_index[local_train]
    test_index = valid_index[local_test]
    overlap = set(authors.iloc[train_index]) & set(authors.iloc[test_index])
    details = {
        "train_authors": int(authors.iloc[train_index].nunique()),
        "test_authors": int(authors.iloc[test_index].nunique()),
        "overlap_authors": int(len(overlap)),
    }
    return train_index, test_index, details


def top_recall(actual: np.ndarray, predicted: np.ndarray, fraction: float = 0.20) -> float:
    count = max(1, int(len(actual) * fraction))
    actual_top = set(np.argpartition(actual, -count)[-count:])
    predicted_top = set(np.argpartition(predicted, -count)[-count:])
    return len(actual_top & predicted_top) / count


def evaluate_split(
    name: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
    features: pd.DataFrame,
    heat: pd.Series,
    high_heat: pd.Series,
) -> list[dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(features.iloc[train_index])
    x_test = imputer.transform(features.iloc[test_index])

    classifier = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    classifier.fit(x_train, high_heat.iloc[train_index])
    class_score = classifier.predict_proba(x_test)[:, 1]
    class_prediction = (class_score >= 0.5).astype(int)
    truth = high_heat.iloc[test_index].to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, class_prediction, labels=[0, 1]).ravel()
    class_row: dict[str, object] = {
        "切分": name,
        "任务": "分类",
        "模型": "梯度提升树",
        "训练样本": int(len(train_index)),
        "测试样本": int(len(test_index)),
        "测试正类比例": float(truth.mean()),
        "ROC-AUC": float(roc_auc_score(truth, class_score)),
        "PR-AUC": float(average_precision_score(truth, class_score)),
        "F1": float(f1_score(truth, class_prediction, zero_division=0)),
        "Precision": float(precision_score(truth, class_prediction, zero_division=0)),
        "Recall": float(recall_score(truth, class_prediction, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    regressor = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    train_heat = heat.iloc[train_index].to_numpy()
    test_heat = heat.iloc[test_index].to_numpy()
    regressor.fit(x_train, np.log1p(train_heat))
    heat_prediction = np.clip(np.expm1(regressor.predict(x_test)), 0, None)
    correlation = spearmanr(test_heat, heat_prediction).statistic
    regression_row: dict[str, object] = {
        "切分": name,
        "任务": "回归",
        "模型": "梯度提升树",
        "训练样本": int(len(train_index)),
        "测试样本": int(len(test_index)),
        "MAE": float(mean_absolute_error(test_heat, heat_prediction)),
        "RMSE": float(mean_squared_error(test_heat, heat_prediction) ** 0.5),
        "RMSLE": float(mean_squared_log_error(test_heat, heat_prediction) ** 0.5),
        "R2": float(r2_score(test_heat, heat_prediction)),
        "Spearman": float(correlation),
        "Top20Recall": float(top_recall(test_heat, heat_prediction)),
    }
    return [class_row, regression_row]


def main() -> None:
    frame = read_detail_csvs()
    features, heat, high_heat, post_date, authors, metadata = prepare(frame)
    time_train, time_test, time_details = temporal_split(post_date)
    group_train, group_test, group_details = author_group_split(authors)

    rows = []
    rows.extend(evaluate_split("时间外切分", time_train, time_test, features, heat, high_heat))
    rows.extend(evaluate_split("按博主分组", group_train, group_test, features, heat, high_heat))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "12_2_robustness_metrics.csv", index=False, encoding="utf-8-sig")

    metadata.update(
        {
            "time_split": {
                "train_rows": int(len(time_train)),
                "test_rows": int(len(time_test)),
                **time_details,
            },
            "author_group_split": {
                "train_rows": int(len(group_train)),
                "test_rows": int(len(group_test)),
                **group_details,
            },
        }
    )
    with open(OUT / "12_1_experiment_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
