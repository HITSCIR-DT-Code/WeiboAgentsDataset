"""
nohup python compute_ppl.py > zz_ppl.log1 2>&1 &
ps -ef | grep compute_ppl.py
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_weibo_groups, load_moltbook_agents
# ── 模型配置 ──────────────────────────────────────────────────────────────────
# 修改此路径指向本地 Qwen2.5 权重目录，或使用 HuggingFace 模型 ID
MODEL_NAME = str(Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-1.5B")

# ── 模型加载（延迟加载：仅在需要推理时加载，读缓存时跳过）──────────────────
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None

def _get_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading PPL model: {MODEL_NAME}  (device={DEVICE}) ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        _model.eval()
    return _tokenizer, _model

MAX_LENGTH = 1024   # 单条文本最大 token 数（降低以节省显存）
BATCH_SIZE = 6      # 批量推理（Qwen3.5-4B 词表大，batch 太大会 OOM）

# ── 配置 ──────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "ppl_cache.json"

GROUPS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]
COLORS = ["limegreen", "tomato", "orange", "goldenrod"]
LABELS = ["Human", "Bot", "Agent\n(Weibo)", "Agent\n(Moltbook)"]
LEGEND_LABELS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]


# ── PPL 计算 ──────────────────────────────────────────────────────────────────
def compute_ppl_batch(texts: list[str]) -> list[float]:
    """批量计算每条文本的困惑度，返回与 texts 等长的 PPL 列表。"""
    tokenizer, model = _get_model()
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        # 过滤空文本
        valid_indices = [j for j, t in enumerate(batch_texts) if t.strip()]
        valid_texts = [batch_texts[j] for j in valid_indices]

        # 初始化 batch 结果（空文本 PPL = NaN）
        batch_ppls = [np.nan] * len(batch_texts)

        if not valid_texts:
            results.extend(batch_ppls)
            continue

        inputs = tokenizer(
            valid_texts, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_LENGTH,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # 释放 outputs 引用，减少显存占用
        logits = outputs.logits  # [batch, seq_len, vocab]
        del outputs

        for idx_in_batch, orig_idx in enumerate(valid_indices):
            # 取该样本的有效 token 数（不含 padding）
            attn = inputs["attention_mask"][idx_in_batch]
            token_len = attn.sum().item()
            if token_len <= 1:
                batch_ppls[orig_idx] = np.nan
                continue
            # 该样本的 logits 和 labels（shift 一位做 next-token prediction）
            sample_logits = logits[idx_in_batch, :token_len - 1, :]  # [T-1, V]
            sample_labels = inputs["input_ids"][idx_in_batch, 1:token_len]  # [T-1]
            loss_fn = torch.nn.CrossEntropyLoss()
            nll = loss_fn(sample_logits, sample_labels).item()
            batch_ppls[orig_idx] = math.exp(nll)

        results.extend(batch_ppls)
    return results


# ── 特征聚合 ──────────────────────────────────────────────────────────────────
def compute_user_ppl(texts: list[str]) -> dict | None:
    """给定帖子文本列表，返回每用户 PPL 聚合指标。"""
    if not texts:
        return None
    ppls = np.array(compute_ppl_batch(texts), dtype=float)
    ppls = ppls[np.isfinite(ppls)]
    if len(ppls) == 0:
        return None
    return {
        "avg_ppl": float(np.mean(ppls)),
        "median_ppl": float(np.median(ppls)),
        "ppl_cv": float(np.std(ppls) / np.mean(ppls)) if np.mean(ppls) > 0 else None,
    }


# ── 缓存 ─────────────────────────────────────────────────────────────────────
def _build_ppl_cache(groups_weibo: dict, agents_moltbook: list) -> dict[str, list]:
    """计算全部用户的 PPL 特征并写入缓存文件，返回 feat dict。"""
    feat: dict[str, list] = {g: [] for g in GROUPS}

    # Weibo 三组
    for g in ("Human", "Bot", "Agent(Weibo)"):
        users = groups_weibo.get(g, [])
        for u in tqdm(users, desc=f"Processing {g}", unit="user"):
            texts = [t["text"] for t in u.get("tweets", []) if t.get("text", "").strip()]
            r = compute_user_ppl(texts)
            if r:
                feat[g].append(r)
        torch.cuda.empty_cache()

    # Moltbook
    for u in tqdm(agents_moltbook, desc="Processing Agent(Moltbook)", unit="user"):
        texts = [p["text"] for p in u.get("posts", []) if p.get("text", "").strip()]
        r = compute_user_ppl(texts)
        if r:
            feat["Agent(Moltbook)"].append(r)
    torch.cuda.empty_cache()

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(feat, f, ensure_ascii=False, indent=2)
    print(f"Cache saved → {CACHE_FILE}  ({sum(len(v) for v in feat.values())} users)")
    return feat


def _load_ppl_cache() -> dict[str, list] | None:
    """从缓存文件加载 PPL 特征。不存在则返回 None。"""
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        feat = json.load(f)
    print(f"Cache loaded → {CACHE_FILE}  ({sum(len(v) for v in feat.values())} users)")
    return feat


def get_ppl_features(groups_weibo: dict, agents_moltbook: list) -> dict[str, list]:
    """获取 PPL 特征（优先读缓存，缓存不存在则推理并写缓存）。"""
    cached = _load_ppl_cache()
    if cached is not None:
        return cached
    print("No cache found. Computing perplexity for all posts (this may take a while)...")
    return _build_ppl_cache(groups_weibo, agents_moltbook)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
METRICS = [
    ("avg_ppl",    "Avg Perplexity",       False),
    ("median_ppl", "Median Perplexity",    True),
    ("ppl_cv",     "Perplexity CV",        False),
]


def _violin(ax, data_groups: list[np.ndarray], group_keys: list[str],
            title: str, ylabel: str, log_scale: bool = False):
    valid_data, valid_colors, valid_keys = [], [], []
    for arr, key in zip(data_groups, group_keys):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        # IQR 过滤极端离群值
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        arr = arr[(arr >= lower) & (arr <= upper)]
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


def plot_ppl(feat: dict[str, list]):
    def _arr(gkey, metric):
        return np.array([f[metric] for f in feat[gkey] if f.get(metric) is not None], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Perplexity Distribution by Account Type",
                 fontsize=13, y=1.01, fontweight="bold")

    for ax, (key, title, use_log) in zip(axes, METRICS):
        data = [_arr(g, key) for g in GROUPS]
        _violin(ax, data, GROUPS, title, title, log_scale=use_log)

    patches = [mpatches.Patch(color=COLORS[i], label=LEGEND_LABELS[i]) for i in range(len(GROUPS))]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=False)

    plt.tight_layout()
    out = PLOTS_DIR / "ppl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    weibo = load_weibo_groups()
    molts = load_moltbook_agents()
    feat = get_ppl_features(weibo, molts)
    print("Plotting perplexity features...")
    plot_ppl(feat)
