from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "clustering.json"

POSITIVE_WORDS = [
    "好吃", "不错", "满意", "推荐", "喜欢", "新鲜", "干净", "热情", "周到", "实惠",
    "便宜", "丰富", "舒服", "方便", "正宗", "美味", "香", "赞", "值得", "放心",
]
NEGATIVE_WORDS = [
    "难吃", "不好", "失望", "差", "贵", "脏", "慢", "冷淡", "排队", "拥挤",
    "不新鲜", "咸", "油腻", "少", "坑", "糟糕", "等很久", "态度恶劣", "卫生差",
]
THEMES = {
    "taste": ["味道", "口味", "好吃", "难吃", "菜品", "正宗", "香", "甜", "咸", "辣"],
    "environment": ["环境", "装修", "氛围", "安静", "吵", "拥挤", "座位", "包间"],
    "service": ["服务", "态度", "服务员", "热情", "冷淡", "周到", "上菜"],
    "value": ["价格", "价钱", "便宜", "贵", "实惠", "性价比", "划算", "优惠"],
    "hygiene": ["卫生", "干净", "脏", "整洁", "餐具", "异物"],
    "wait": ["排队", "等位", "等很久", "上菜慢", "速度", "预约"],
    "portion": ["分量", "份量", "量大", "量少", "份大", "份小"],
    "fresh": ["新鲜", "不新鲜", "食材", "现做", "冷冻", "日期"],
}

RATING_FIELDS = ["rating", "rating_env", "rating_flavor", "rating_service"]
STAT_NAMES = ["review_count"]
for _field in RATING_FIELDS:
    STAT_NAMES.extend([f"{_field}_n", f"{_field}_sum", f"{_field}_sq"])
STAT_NAMES.extend(["comment_count", "positive_review_count", "negative_review_count"])
STAT_NAMES.extend([f"theme_{name}_count" for name in THEMES])
STAT_INDEX = {name: i for i, name in enumerate(STAT_NAMES)}


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def load_config(path: Path = CONFIG_PATH) -> dict:
    """读取项目配置。"""
    return json.loads(path.read_text(encoding="utf-8"))


def compile_pattern(words: Iterable[str]) -> re.Pattern[str]:
    """将关键词编译为安全的正则表达式。"""
    return re.compile("|".join(sorted((re.escape(word) for word in words), key=len, reverse=True)))


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    """计算文件首尾区块指纹，兼顾大文件速度和可追溯性。"""
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(block_size))
        if size > block_size:
            handle.seek(max(0, size - block_size))
            digest.update(handle.read(block_size))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def read_restaurants(path: Path) -> pd.DataFrame:
    """读取商家主表并校验主键。"""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "restId" not in frame.columns:
        raise ValueError("restaurants.csv 缺少 restId 字段")
    if frame["restId"].eq("").any():
        raise ValueError("restaurants.csv 存在空 restId")
    if frame["restId"].duplicated().any():
        raise ValueError("restaurants.csv 存在重复 restId")
    return frame


def aggregate_ratings(
    ratings_path: Path,
    restaurants: pd.DataFrame,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    """分块扫描评价表，累计商家级评分和文本统计。"""
    id_to_index = {value: i for i, value in enumerate(restaurants["restId"].tolist())}
    stats = np.zeros((len(restaurants), len(STAT_NAMES)), dtype=np.float64)
    positive_pattern = compile_pattern(POSITIVE_WORDS)
    negative_pattern = compile_pattern(NEGATIVE_WORDS)
    theme_patterns = {name: compile_pattern(words) for name, words in THEMES.items()}
    total_rows = 0
    orphan_rows = 0
    invalid_ratings = {field: 0 for field in RATING_FIELDS}
    duplicate_rows_within_chunks = 0
    started = time.time()

    reader = pd.read_csv(
        ratings_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=chunk_size,
        on_bad_lines="skip",
    )
    required = {"restId", "comment", *RATING_FIELDS}

    for chunk_number, chunk in enumerate(reader, start=1):
        if not required.issubset(chunk.columns):
            missing = sorted(required.difference(chunk.columns))
            raise ValueError(f"ratings.csv 缺少字段: {missing}")
        total_rows += len(chunk)
        if {"userId", "timestamp"}.issubset(chunk.columns):
            duplicate_rows_within_chunks += int(
                chunk.duplicated(subset=["userId", "restId", "timestamp"], keep="first").sum()
            )

        mapped = chunk["restId"].map(id_to_index)
        orphan_rows += int(mapped.isna().sum())
        valid_link = mapped.notna().to_numpy()
        if not valid_link.any():
            continue
        row_indices = mapped[valid_link].astype(np.int64).to_numpy()
        linked = chunk.loc[valid_link].copy()
        matrix = np.zeros((len(linked), len(STAT_NAMES)), dtype=np.float64)
        matrix[:, STAT_INDEX["review_count"]] = 1.0

        for field in RATING_FIELDS:
            raw = pd.to_numeric(linked[field], errors="coerce")
            invalid = raw.notna() & ~raw.between(1.0, 5.0)
            invalid_ratings[field] += int(invalid.sum())
            values = raw.where(raw.between(1.0, 5.0)).to_numpy(dtype=np.float64, na_value=np.nan)
            present = ~np.isnan(values)
            matrix[:, STAT_INDEX[f"{field}_n"]] = present
            matrix[:, STAT_INDEX[f"{field}_sum"]] = np.nan_to_num(values)
            matrix[:, STAT_INDEX[f"{field}_sq"]] = np.nan_to_num(values * values)

        comments = linked["comment"].astype(str)
        has_comment = comments.str.strip().ne("")
        matrix[:, STAT_INDEX["comment_count"]] = has_comment.to_numpy(dtype=np.float64)
        matrix[:, STAT_INDEX["positive_review_count"]] = comments.str.contains(
            positive_pattern, regex=True, na=False
        ).to_numpy(dtype=np.float64)
        matrix[:, STAT_INDEX["negative_review_count"]] = comments.str.contains(
            negative_pattern, regex=True, na=False
        ).to_numpy(dtype=np.float64)
        for name, pattern in theme_patterns.items():
            matrix[:, STAT_INDEX[f"theme_{name}_count"]] = comments.str.contains(
                pattern, regex=True, na=False
            ).to_numpy(dtype=np.float64)

        np.add.at(stats, row_indices, matrix)
        if chunk_number % 10 == 0:
            elapsed = max(time.time() - started, 0.001)
            print(f"已处理 {total_rows:,} 条评价，速度 {total_rows / elapsed:,.0f} 条/秒", flush=True)

    quality = {
        "rating_rows_processed": int(total_rows),
        "orphan_rating_rows": int(orphan_rows),
        "duplicate_rows_within_chunks": int(duplicate_rows_within_chunks),
        "invalid_rating_values": invalid_ratings,
    }
    return stats, quality


def smoothed_rate(counts: np.ndarray, totals: np.ndarray, prior_weight: float) -> np.ndarray:
    total_count = float(totals.sum())
    global_rate = float(counts.sum() / total_count) if total_count else 0.0
    return (counts + prior_weight * global_rate) / (totals + prior_weight)


def build_features(stats: np.ndarray, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """由累计统计构建可解释商家级特征。"""
    # 评分先验约等于补充若干条全局均值评价，避免小样本均值过度极端。
    prior = float(config["rating_prior_weight"])
    # 文本先验单独设置，因为关键词比例比评分均值更容易受短评论影响。
    text_prior = float(config["text_prior_weight"])
    result: dict[str, np.ndarray] = {
        "review_count": stats[:, STAT_INDEX["review_count"]],
        "comment_count": stats[:, STAT_INDEX["comment_count"]],
    }

    raw_means: dict[str, np.ndarray] = {}
    smooth_means: dict[str, np.ndarray] = {}
    standard_deviations = []
    for field in RATING_FIELDS:
        n = stats[:, STAT_INDEX[f"{field}_n"]]
        sums = stats[:, STAT_INDEX[f"{field}_sum"]]
        squares = stats[:, STAT_INDEX[f"{field}_sq"]]
        global_mean = float(sums.sum() / n.sum()) if n.sum() else 0.0
        raw = np.divide(sums, n, out=np.full_like(sums, global_mean), where=n > 0)
        smooth = (sums + prior * global_mean) / (n + prior)
        variance = np.divide(squares, n, out=np.zeros_like(squares), where=n > 0) - raw * raw
        standard_deviations.append(np.sqrt(np.maximum(variance, 0.0)))
        raw_means[field] = raw
        smooth_means[field] = smooth
        result[f"{field}_count"] = n
        result[f"{field}_mean_raw"] = raw

    result["overall_score"] = smooth_means["rating"]
    result["environment_score"] = smooth_means["rating_env"]
    result["flavor_score"] = smooth_means["rating_flavor"]
    result["service_score"] = smooth_means["rating_service"]
    dimension_average = (
        result["environment_score"] + result["flavor_score"] + result["service_score"]
    ) / 3.0
    result["environment_gap"] = result["environment_score"] - dimension_average
    result["flavor_gap"] = result["flavor_score"] - dimension_average
    result["service_gap"] = result["service_score"] - dimension_average
    result["rating_volatility"] = np.mean(np.vstack(standard_deviations), axis=0)

    comments = result["comment_count"]
    positive = stats[:, STAT_INDEX["positive_review_count"]]
    negative = stats[:, STAT_INDEX["negative_review_count"]]
    positive_rate = smoothed_rate(positive, comments, text_prior)
    negative_rate = smoothed_rate(negative, comments, text_prior)
    result["positive_rate"] = positive_rate
    result["negative_rate"] = negative_rate
    result["sentiment_balance"] = positive_rate - negative_rate
    for name in THEMES:
        counts = stats[:, STAT_INDEX[f"theme_{name}_count"]]
        result[f"theme_{name}_rate"] = smoothed_rate(counts, comments, text_prior)

    frame = pd.DataFrame(result)
    feature_names = list(config["feature_weights"].keys())
    return frame, feature_names


def standardize_features(
    frame: pd.DataFrame, feature_names: list[str], eligible: np.ndarray, weights: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.loc[eligible, feature_names].to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, medians)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-9] = 1.0
    standardized = np.clip((values - means) / scales, -4.0, 4.0)
    # 评分是主信号，主题命中率只提供辅助解释，权重在配置文件中集中维护。
    weight_vector = np.array([float(weights[name]) for name in feature_names], dtype=np.float64)
    return standardized * weight_vector, means, scales


def initialize_centers(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(0, len(x))]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for index in range(1, k):
        total = closest.sum()
        choice = rng.integers(0, len(x)) if total <= 0 else rng.choice(len(x), p=closest / total)
        centers[index] = x[choice]
        closest = np.minimum(closest, np.sum((x - centers[index]) ** 2, axis=1))
    return centers


def assign_clusters(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * x @ centers.T
    )
    labels = np.argmin(distances, axis=1)
    minimum = distances[np.arange(len(x)), labels]
    return labels, minimum


def run_kmeans(
    x: np.ndarray,
    k: int,
    seed: int,
    restarts: int,
    max_iterations: int,
) -> KMeansResult:
    """无额外依赖的确定性 K-means 实现。"""
    best: KMeansResult | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart * 104729)
        centers = initialize_centers(x, k, rng)
        labels = np.zeros(len(x), dtype=np.int32)
        for _ in range(max_iterations):
            new_labels, minimum = assign_clusters(x, centers)
            new_centers = centers.copy()
            for cluster in range(k):
                members = x[new_labels == cluster]
                if len(members) == 0:
                    new_centers[cluster] = x[rng.integers(0, len(x))]
                else:
                    new_centers[cluster] = members.mean(axis=0)
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            labels = new_labels
            if shift < 1e-5:
                break
        labels, minimum = assign_clusters(x, centers)
        candidate = KMeansResult(labels=labels, centers=centers, inertia=float(minimum.sum()))
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    assert best is not None
    return best


def approximate_silhouette(x: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    """在固定样本上计算轮廓系数，控制大数据内存消耗。"""
    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        indices = rng.choice(len(x), sample_size, replace=False)
        sample = x[indices]
        sample_labels = labels[indices]
    else:
        sample = x
        sample_labels = labels
    if len(np.unique(sample_labels)) < 2:
        return -1.0
    squared = np.maximum(
        np.sum(sample * sample, axis=1, keepdims=True)
        + np.sum(sample * sample, axis=1)[None, :]
        - 2.0 * sample @ sample.T,
        0.0,
    )
    distances = np.sqrt(squared)
    silhouettes = np.zeros(len(sample), dtype=np.float64)
    unique = np.unique(sample_labels)
    for i, cluster in enumerate(sample_labels):
        same = sample_labels == cluster
        same_count = int(same.sum())
        if same_count <= 1:
            silhouettes[i] = 0.0
            continue
        a = float(distances[i, same].sum() / (same_count - 1))
        b = min(
            float(distances[i, sample_labels == other].mean())
            for other in unique
            if other != cluster
        )
        silhouettes[i] = (b - a) / max(a, b, 1e-12)
    return float(silhouettes.mean())


def align_labels(base_centers: np.ndarray, other: KMeansResult) -> np.ndarray:
    k = len(base_centers)
    cost = np.sqrt(np.maximum(
        np.sum(base_centers * base_centers, axis=1, keepdims=True)
        + np.sum(other.centers * other.centers, axis=1)[None, :]
        - 2.0 * base_centers @ other.centers.T,
        0.0,
    ))
    best_perm = min(
        itertools.permutations(range(k)),
        key=lambda permutation: sum(cost[i, permutation[i]] for i in range(k)),
    )
    inverse = np.empty(k, dtype=np.int32)
    for base_label, other_label in enumerate(best_perm):
        inverse[other_label] = base_label
    return inverse[other.labels]


def select_k(x: np.ndarray, config: dict) -> tuple[KMeansResult, pd.DataFrame, float]:
    """比较候选聚类数并计算多随机种子稳定性。"""
    # 固定种子保证候选比较、稳定性测试和正式输出均可复现。
    seed = int(config["random_seed"])
    diagnostics = []
    results: dict[int, KMeansResult] = {}
    for k in config["candidate_k"]:
        result = run_kmeans(
            x, int(k), seed, int(config["kmeans_restarts"]), int(config["kmeans_max_iterations"])
        )
        counts = np.bincount(result.labels, minlength=int(k))
        silhouette = approximate_silhouette(
            x, result.labels, int(config["silhouette_sample_size"]), seed + int(k)
        )
        minimum_share = float(counts.min() / len(x))
        adjusted = silhouette - (0.03 if minimum_share < 0.02 else 0.0)
        diagnostics.append({
            "k": int(k),
            "silhouette": silhouette,
            "minimum_cluster_share": minimum_share,
            "inertia_per_restaurant": result.inertia / len(x),
            "selection_score": adjusted,
        })
        results[int(k)] = result
        print(f"候选 k={k}: silhouette={silhouette:.4f}, 最小簇占比={minimum_share:.2%}", flush=True)
    diagnostic_frame = pd.DataFrame(diagnostics).sort_values("k")
    ranked = diagnostic_frame.sort_values(
        ["selection_score", "minimum_cluster_share"],
        ascending=False,
    )
    best_k = int(ranked.iloc[0]["k"])
    best = results[best_k]
    agreements = []
    for offset in (1, 2):
        other = run_kmeans(
            x,
            best_k,
            seed + offset * 1009,
            max(2, int(config["kmeans_restarts"]) // 2),
            int(config["kmeans_max_iterations"]),
        )
        aligned = align_labels(best.centers, other)
        agreements.append(float(np.mean(aligned == best.labels)))
    stability = float(np.mean(agreements)) if agreements else 1.0
    return best, diagnostic_frame, stability


def choose_business_labels(profile: pd.DataFrame) -> dict[int, str]:
    """根据聚类中心的相对特征生成唯一、可读的业务标签。"""
    metric_columns = [
        "overall_score", "environment_score", "flavor_score", "service_score",
        "rating_volatility", "sentiment_balance", "negative_rate",
    ]
    z = profile[metric_columns].copy()
    for column in metric_columns:
        std = float(z[column].std(ddof=0))
        z[column] = (z[column] - z[column].mean()) / (std if std > 1e-9 else 1.0)
    labels: dict[int, str] = {}
    used: set[str] = set()
    for cluster in profile.sort_values("overall_score", ascending=False)["cluster_id"].astype(int):
        row = profile.loc[profile["cluster_id"] == cluster].iloc[0]
        zr = z.loc[profile["cluster_id"] == cluster].iloc[0]
        gaps = {
            "口味优势型": row["flavor_score"] - (row["environment_score"] + row["service_score"]) / 2,
            "环境优势型": row["environment_score"] - (row["flavor_score"] + row["service_score"]) / 2,
            "服务优势型": row["service_score"] - (row["environment_score"] + row["flavor_score"]) / 2,
        }
        all_dimensions_above_average = min(
            zr["environment_score"],
            zr["flavor_score"],
            zr["service_score"],
        ) >= 0
        if zr["overall_score"] >= 0.8 and all_dimensions_above_average:
            candidates = ["全面高口碑型", max(gaps, key=gaps.get)]
        elif zr["overall_score"] <= -0.8 and zr["negative_rate"] >= 0:
            candidates = ["整体改善型", "负面反馈集中型"]
        elif zr["rating_volatility"] >= 0.8:
            candidates = ["评价分化型"]
        # 只有维度差异足够大时才称为“优势型”，避免把轻微偏高包装成显著优势。
        elif max(gaps.values()) >= 0.25:
            candidates = [max(gaps, key=gaps.get)]
        elif zr["sentiment_balance"] >= 0.6:
            candidates = ["口碑认可型"]
        elif zr["negative_rate"] >= 0.6:
            candidates = ["体验短板型"]
        else:
            candidates = ["均衡稳定型"]
        candidates.extend(["口味优势型", "环境优势型", "服务优势型", "评价分化型", "均衡稳定型"])
        selected = next((name for name in candidates if name not in used), f"差异特征型{cluster + 1}")
        labels[cluster] = selected
        used.add(selected)
    return labels


def build_cluster_profiles(
    features: pd.DataFrame,
    eligible: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    profiled = features.loc[eligible].copy()
    profiled["cluster_id"] = labels
    numeric = [
        column
        for column in features.columns
        if column not in {"review_count", "comment_count"}
    ]
    profile = profiled.groupby("cluster_id")[numeric].mean().reset_index()
    counts = profiled.groupby("cluster_id").size().rename("restaurant_count")
    review_counts = profiled.groupby("cluster_id")["review_count"].sum().rename("review_count")
    profile = profile.join(counts, on="cluster_id").join(review_counts, on="cluster_id")
    profile["cluster_share"] = profile["restaurant_count"] / len(profiled)
    return profile


def explain_cluster(row: pd.Series, overall: pd.Series, label: str) -> tuple[str, str, str]:
    dimensions = {
        "口味": row["flavor_score"],
        "环境": row["environment_score"],
        "服务": row["service_score"],
    }
    strongest = max(dimensions, key=dimensions.get)
    weakest = min(dimensions, key=dimensions.get)
    score_delta = row["overall_score"] - overall["overall_score"]
    sentiment_delta = row["sentiment_balance"] - overall["sentiment_balance"]
    dimension_span = max(dimensions.values()) - min(dimensions.values())
    if label == "全面高口碑型":
        reason = (
            f"总体平滑评分较均值高 {score_delta:.2f} 分，口味、环境和服务均处于四类最高水平；"
            f"环境和服务主题评论占比分别为 {row['theme_environment_rate']:.1%} 和 {row['theme_service_rate']:.1%}。"
        )
        distinction = "该类不是单一维度突出，而是总体及三个细分评分同步领先。"
        advice = "保持综合体验，并重点控制评分波动，排查门店、时段或人员造成的不一致。"
    elif label == "口味优势型":
        reason = (
            f"口味评分为 {row['flavor_score']:.2f}，分别高于环境和服务 "
            f"{row['flavor_score'] - row['environment_score']:.2f}、"
            f"{row['flavor_score'] - row['service_score']:.2f} 分；"
            f"{row['theme_taste_rate']:.1%} 的有评论评价提及口味相关词，是四类中最高。"
        )
        distinction = "口味形成清晰主优势，但环境和服务没有同步达到同等水平。"
        advice = "保持核心菜品稳定，把环境和服务作为优先补齐项。"
    elif label == "均衡稳定型":
        volatility_delta = overall["rating_volatility"] - row["rating_volatility"]
        reason = (
            f"总体平滑评分与均值仅相差 {abs(score_delta):.2f} 分，三个细分评分最大差距为 "
            f"{dimension_span:.2f} 分；评分波动较均值低 {max(volatility_delta, 0):.2f}。"
        )
        distinction = "没有单项形成压倒性优势或明显短板，评价结构更接近整体中位状态。"
        advice = "维持现有稳定性，优先从服务细节和复购体验寻找增量。"
    elif label == "整体改善型":
        reason = (
            f"总体平滑评分较均值低 {abs(score_delta):.2f} 分，口味、环境和服务均偏低；"
            f"负面词评价占比 {row['negative_rate']:.1%}，情感净值较均值低 {abs(sentiment_delta):.1%}。"
        )
        distinction = "与单项短板类别不同，该类的低分分布在多个体验维度。"
        advice = f"先处理{weakest}维度及卫生、等待等高频问题，再跟踪整改后的差评率。"
    else:
        reason = (
            f"总体评分较均值{'高' if score_delta >= 0 else '低'} {abs(score_delta):.2f} 分；"
            f"{strongest}相对较强，{weakest}相对较弱。"
        )
        distinction = "根据总体评分、维度结构和评论情感与其他类别区分。"
        advice = f"保持{strongest}表现，并优先改善{weakest}。"
    return reason, distinction, advice


def write_outputs(
    restaurants: pd.DataFrame,
    features: pd.DataFrame,
    eligible: np.ndarray,
    cluster_result: KMeansResult,
    profiles: pd.DataFrame,
    label_map: dict[int, str],
    diagnostics: pd.DataFrame,
    stability: float,
    quality: dict,
    config: dict,
) -> None:
    outputs = ROOT / config["paths"]["outputs"]
    reports = ROOT / config["paths"]["reports"]
    docs = ROOT / config["paths"]["docs"]
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    minimum_reviews = int(config["minimum_reviews_for_clustering"])
    labels = np.full(len(restaurants), "样本不足", dtype=object)
    labels[features["review_count"].to_numpy() == 0] = "暂无有效评价"
    eligible_indices = np.flatnonzero(eligible)
    labels[eligible_indices] = [label_map[int(value)] for value in cluster_result.labels]

    labeled = restaurants.copy()
    labeled["类别标签"] = labels
    output_csv = outputs / "restaurants_labeled.csv"
    labeled.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    evidence = pd.concat(
        [restaurants[["restId"]].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    evidence["类别标签"] = labels
    evidence.to_csv(reports / "merchant_evidence.csv", index=False, encoding="utf-8-sig")

    profile_out = profiles.copy()
    profile_out.insert(1, "类别标签", profile_out["cluster_id"].map(label_map))
    profile_out.to_csv(reports / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(reports / "cluster_diagnostics.csv", index=False, encoding="utf-8-sig")

    quality.update({
        "restaurant_rows": int(len(restaurants)),
        "restaurants_with_reviews": int((features["review_count"] > 0).sum()),
        "restaurants_eligible_for_clustering": int(eligible.sum()),
        "restaurants_sample_insufficient": int(((features["review_count"] > 0) & ~eligible).sum()),
        "restaurants_without_reviews": int((features["review_count"] == 0).sum()),
        "empty_restaurant_names": int(restaurants.get("name", pd.Series(dtype=str)).eq("").sum()),
        "selected_k": int(len(profiles)),
        "cluster_stability_agreement": stability,
        "minimum_reviews_for_clustering": minimum_reviews,
    })
    (reports / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = features.loc[eligible].mean(numeric_only=True)
    lines = [
        "# 口碑分类说明", "",
        "## 使用说明", "",
        "分类综合总体评分、环境、口味、服务、评分波动、评论情感和主题信号。标签反映数据中的相对口碑结构，不是绝对质量认证。", "",
        f"有效评价少于 {minimum_reviews} 条的商家标记为 `样本不足`；无关联评价的商家标记为 `暂无有效评价`。", "",
        "## 标签定义", "",
    ]
    for _, row in profile_out.sort_values("类别标签").iterrows():
        reason, distinction, advice = explain_cluster(
            row,
            overall,
            str(row["类别标签"]),
        )
        lines.extend([
            f"### {row['类别标签']}", "",
            f"- 商家数：{int(row['restaurant_count']):,}，占可聚类商家的 {row['cluster_share']:.1%}。",
            (
                f"- 评分特征：总体 {row['overall_score']:.2f}，"
                f"口味 {row['flavor_score']:.2f}，环境 {row['environment_score']:.2f}，"
                f"服务 {row['service_score']:.2f}。"
            ),
            f"- 分类原因：{reason}",
            f"- 类别区别：{distinction}",
            f"- 经营建议：{advice}", "",
        ])
    lines.extend([
        "### 样本不足", "",
        f"已有评价但少于 {minimum_reviews} 条。信息不足以稳定判断口碑结构，保留为独立标签，等待后续数据积累。", "",
        "### 暂无有效评价", "",
        "商家主表中存在，但评价表中没有可关联记录。该标签表示缺少证据，不代表好评或差评。", "",
    ])
    (docs / "口碑分类说明.md").write_text("\n".join(lines), encoding="utf-8")

    distribution = labeled["类别标签"].value_counts()
    selected = diagnostics.sort_values("selection_score", ascending=False).iloc[0]
    report = [
        "# 餐馆多维度口碑聚类项目报告", "",
        "## 项目结论", "",
        (
            f"本次对 {len(restaurants):,} 家商家进行处理，其中 "
            f"{int(eligible.sum()):,} 家达到聚类最低样本要求。最终选择 "
            f"{len(profiles)} 个算法类别，另设 `样本不足` 和 `暂无有效评价` "
            "两个证据状态标签。"
        ),
        "",
        "主输出保持原商家字段和值，仅新增 `类别标签`。完整商家级特征保存在审计报告中。", "",
        "## 数据质量", "",
        f"- 成功解析评价记录：{quality['rating_rows_processed']:,} 条。",
        f"- 无法关联商家的评价：{quality['orphan_rating_rows']:,} 条。",
        f"- 商家名称为空：{quality['empty_restaurant_names']:,} 家。",
        f"- 可聚类商家：{quality['restaurants_eligible_for_clustering']:,} 家。",
        f"- 样本不足商家：{quality['restaurants_sample_insufficient']:,} 家。",
        f"- 无评价商家：{quality['restaurants_without_reviews']:,} 家。", "",
        "重复记录统计仅覆盖同一数据分块内部的 `userId + restId + timestamp` 重复，属于保守下界，不作为自动删重依据。", "",
        "## 方法", "",
        (
            "评分特征使用全局先验平滑，降低小样本极端值影响。评论特征使用中文领域词表"
            "识别正负倾向及口味、环境、服务、价格、卫生、排队、分量和新鲜度主题。"
            "主题命中只代表讨论强度，不直接代表褒贬。"
        ),
        "",
        "各特征在可聚类商家中标准化并按配置加权。候选聚类数为 4 至 8，采用固定样本轮廓系数、最小簇占比和多随机种子一致率选择。", "",
        "## 模型选择", "",
        (
            f"选择 k={int(selected['k'])}，样本轮廓系数 "
            f"{selected['silhouette']:.4f}，最小簇占比 "
            f"{selected['minimum_cluster_share']:.1%}，多随机种子标签一致率 "
            f"{stability:.1%}。"
        ),
        "",
        "候选结果：", "",
        "| k | 轮廓系数 | 最小簇占比 | 每商家惯性 | 选择分 |", "|---:|---:|---:|---:|---:|",
    ]
    for _, row in diagnostics.iterrows():
        report.append(
            f"| {int(row['k'])} | {row['silhouette']:.4f} | "
            f"{row['minimum_cluster_share']:.1%} | "
            f"{row['inertia_per_restaurant']:.3f} | "
            f"{row['selection_score']:.4f} |"
        )
    report.extend(["", "## 分类结果", "", "| 类别标签 | 商家数 | 占全部商家 |", "|---|---:|---:|"])
    for name, count in distribution.items():
        report.append(f"| {name} | {count:,} | {count / len(labeled):.1%} |")
    report.extend([
        "", "## 局限性", "",
        "- 商家名称缺失时只能依靠 `restId` 识别。",
        "- 词表法适合大规模、可复现的主题信号提取，但不能完整理解反讽、否定范围和复杂语境。",
        "- 聚类描述的是当前数据中的相对结构，不能直接解释为因果关系或官方评级。",
        "- 历史评论的时间分布可能不均，若用于当前经营决策，应补充近期数据并做时间衰减。", "",
        "## 复现", "",
        (
            "在项目根目录执行 `python -m src.pipeline`，再执行 "
            "`python -m unittest discover -s tests -v`。所有阈值、随机种子和"
            "特征权重位于 `config/clustering.json`。"
        ),
        "",
    ])
    (docs / "项目报告.md").write_text("\n".join(report), encoding="utf-8")


def run(config_path: Path = CONFIG_PATH) -> None:
    config = load_config(config_path)
    restaurants_path = ROOT / config["paths"]["restaurants"]
    ratings_path = ROOT / config["paths"]["ratings"]
    restaurants = read_restaurants(restaurants_path)
    print(f"商家数: {len(restaurants):,}", flush=True)
    stats, quality = aggregate_ratings(ratings_path, restaurants, int(config["chunk_size"]))
    quality["restaurants_fingerprint"] = file_fingerprint(restaurants_path)
    quality["ratings_fingerprint"] = file_fingerprint(ratings_path)
    features, feature_names = build_features(stats, config)
    # 最低评价数是可信度门槛；至少还要存在一个有效评分维度。
    minimum_reviews = int(config["minimum_reviews_for_clustering"])
    rating_count_columns = [
        "rating_count",
        "rating_env_count",
        "rating_flavor_count",
        "rating_service_count",
    ]
    eligible = (features["review_count"].to_numpy() >= minimum_reviews) & (
        features[rating_count_columns]
        .sum(axis=1)
        .to_numpy() > 0
    )
    if int(eligible.sum()) < max(config["candidate_k"]) * 5:
        raise ValueError("达到最低样本要求的商家过少，无法进行稳定聚类")
    x, _, _ = standardize_features(
        features, feature_names, eligible, config["feature_weights"]
    )
    result, diagnostics, stability = select_k(x, config)
    profiles = build_cluster_profiles(features, eligible, result.labels)
    label_map = choose_business_labels(profiles)
    write_outputs(
        restaurants, features, eligible, result, profiles, label_map,
        diagnostics, stability, quality, config,
    )
    print("聚类和报告生成完成", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="餐馆多维度口碑聚类")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
