"""
nohup python compute_sentiment.py > zz_sentiment.log1 2>&1 &
ps -ef | grep compute_sentiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_weibo_groups, load_moltbook_agents

# ── 模型 ──────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = Path(__file__).parent.parent / "models" / "multilingual-sentiment-analysis "
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

# 模型延迟加载（仅在需要推理时加载，读缓存时跳过）
_tokenizer = None
_model = None

def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        _model.eval()
    return _tokenizer, _model

SENTIMENT_MAP = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
SCORE_MAP = {"Very Negative": -2, "Negative": -1, "Neutral": 0, "Positive": 1, "Very Positive": 2}

# ── 配置 ──────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "sentiment_cache.json"

GROUPS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]
COLORS = ["limegreen", "tomato", "orange", "goldenrod"]
LABELS = ["Human", "Bot", "Agent\n(Weibo)", "Agent\n(Moltbook)"]
LEGEND_LABELS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]

BATCH_SIZE = 64  # 批量推理


# ── 情感预测 ──────────────────────────────────────────────────────────────────
def predict_sentiments(texts: list[str]) -> list[str]:
    """批量预测情感标签，返回与 texts 等长的标签列表。"""
    tokenizer, model = _get_model()
    results = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(texts), BATCH_SIZE), total=total_batches,
                  desc="Predicting sentiments", unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        # 过滤空文本
        valid_indices = [j for j, t in enumerate(batch) if t.strip()]
        valid_texts = [batch[j] for j in valid_indices]
        if not valid_texts:
            results.extend(["Neutral"] * len(batch))
            continue

        inputs = tokenizer(
            valid_texts, return_tensors="pt", truncation=True,
            padding=True, max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).tolist()

        # 还原到原 batch 位置
        batch_results = ["Neutral"] * len(batch)
        for idx, p in zip(valid_indices, preds):
            batch_results[idx] = SENTIMENT_MAP[p]
        results.extend(batch_results)
    return results


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon 熵（bits）。"""
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── 特征计算 ──────────────────────────────────────────────────────────────────
def compute_sentiment_features(texts: list[str]) -> dict | None:
    """给定帖子文本列表，返回每用户情感聚合指标。"""
    if not texts:
        return None
    labels = predict_sentiments(texts)
    scores = np.array([SCORE_MAP[l] for l in labels], dtype=float)
    n = len(scores)

    # 5 类分布
    cat_counts = np.zeros(5, dtype=float)
    for l in labels:
        for k, v in SENTIMENT_MAP.items():
            if v == l:
                cat_counts[k] += 1
                break

    return {
        "avg_sentiment":    float(np.mean(scores)),
        "positive_ratio":   float(np.sum(scores > 0) / n),
        "negative_ratio":   float(np.sum(scores < 0) / n),
        "sentiment_entropy": _shannon_entropy(cat_counts),
    }


# ── 缓存 ─────────────────────────────────────────────────────────────────────
def _build_sentiment_cache(groups_weibo: dict, agents_moltbook: list) -> dict[str, list]:
    """计算全部用户的情感特征并写入缓存文件，返回 feat dict。"""
    feat: dict[str, list] = {g: [] for g in GROUPS}

    # Weibo 三组
    for g in ("Human", "Bot", "Agent(Weibo)"):
        users = groups_weibo.get(g, [])
        for u in tqdm(users, desc=f"Processing {g}", unit="user"):
            texts = [t["text"] for t in u.get("tweets", []) if t.get("text", "").strip()]
            r = compute_sentiment_features(texts)
            if r:
                feat[g].append(r)

    # Moltbook
    for u in tqdm(agents_moltbook, desc="Processing Agent(Moltbook)", unit="user"):
        texts = [p["text"] for p in u.get("posts", []) if p.get("text", "").strip()]
        r = compute_sentiment_features(texts)
        if r:
            feat["Agent(Moltbook)"].append(r)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(feat, f, ensure_ascii=False, indent=2)
    print(f"Cache saved → {CACHE_FILE}  ({sum(len(v) for v in feat.values())} users)")
    return feat


def _load_sentiment_cache() -> dict[str, list] | None:
    """从缓存文件加载情感特征。不存在则返回 None。"""
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        feat = json.load(f)
    print(f"Cache loaded → {CACHE_FILE}  ({sum(len(v) for v in feat.values())} users)")
    return feat


def get_sentiment_features(groups_weibo: dict, agents_moltbook: list) -> dict[str, list]:
    """获取情感特征（优先读缓存，缓存不存在则推理并写缓存）。"""
    cached = _load_sentiment_cache()
    if cached is not None:
        return cached
    print("No cache found. Loading sentiment model & predicting...")
    return _build_sentiment_cache(groups_weibo, agents_moltbook)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
METRICS = [
    ("avg_sentiment",     "Avg Sentiment Score",       False),
    ("positive_ratio",    "Positive Ratio",            False),
    ("negative_ratio",    "Negative Ratio",            False),
    ("sentiment_entropy", "Sentiment Entropy (bits)",  False),
]


def _violin(ax, data_groups: list[np.ndarray], group_keys: list[str],
            title: str, ylabel: str, log_scale: bool = False):
    valid_data, valid_colors, valid_keys = [], [], []
    for arr, key in zip(data_groups, group_keys):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        valid_data.append(arr)
        group_index = GROUPS.index(key)
        valid_colors.append(COLORS[group_index])
        valid_keys.append(group_index)

    if not valid_data:
        ax.set_visible(False)
        return

    positions = np.arange(1, len(valid_data) + 1)
    parts = ax.violinplot(valid_data, positions=positions,
                          showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], valid_colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("white")
        body.set_linewidth(0.8)

    for pos, arr in zip(positions, valid_data):
        med = np.median(arr)
        ax.hlines(med, pos - 0.15, pos + 0.15,
                  colors="black", linewidths=1.1, zorder=5)

    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    if ymax == ymin:
        ymax = ymin + 1
    span = ymax - ymin
    for pos, arr in zip(positions, valid_data):
        ax.text(pos, ymax + span * 0.02,
                f"n={len(arr)}", ha="center", va="bottom",
                fontsize=7.5, color="gray")
    ax.set_ylim(ymin, ymax + span * 0.12)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[i] for i in valid_keys],
                       rotation=0, ha="center", fontsize=8.5)
    ax.set_title(title, fontsize=10, pad=6)
    lbl = ylabel + "\n(log1p scale)" if log_scale else ylabel
    ax.set_ylabel(lbl, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_sentiment(feat: dict[str, list]):
    def _arr(gkey, metric):
        return np.array([f[metric] for f in feat[gkey] if f.get(metric) is not None], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.suptitle("Sentiment Feature Distribution by Account Type",
                 fontsize=13, y=1.01, fontweight="bold")

    for ax, (key, title, use_log) in zip(axes, METRICS):
        data = [_arr(g, key) for g in GROUPS]
        _violin(ax, data, GROUPS, title, title, log_scale=use_log)

    # 第一张图 (avg_sentiment): y 轴刻度换成英文情感标签
    ymin0, ymax0 = axes[0].get_ylim()
    axes[0].set_ylim(min(ymin0, -2.2), max(ymax0, 2.2))
    axes[0].set_yticks([-2, -1, 0, 1, 2])
    axes[0].set_yticklabels(
        ["Very\nNegative", "Negative", "Neutral", "Positive", "Very\nPositive"],
        ha="right", fontsize=7.5)
    # axes[0].tick_params(axis="y", pad=6)

    patches = [mpatches.Patch(color=COLORS[i], label=LEGEND_LABELS[i]) for i in range(len(GROUPS))]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=False)

    plt.tight_layout()
    out = PLOTS_DIR / "sentiment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    weibo = load_weibo_groups()
    molts = load_moltbook_agents()
    feat = get_sentiment_features(weibo, molts)
    print("Plotting sentiment features...")
    plot_sentiment(feat)
