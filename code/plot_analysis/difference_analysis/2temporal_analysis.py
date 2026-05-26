"""
python 2temporal_analysis.py
====================
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_weibo_groups, load_moltbook_agents

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── 统一字体大小 ─────────────────────────────────────────────────────────────
SUPTITLE_SIZE      = 24
SUBPLOT_TITLE_SIZE = 20
AXIS_LABEL_SIZE    = 16
TICK_LABEL_SIZE    = 16
YTICK_LABEL_SIZE   = 14
N_LABEL_SIZE       = 7.5
LEGEND_SIZE        = 12

# ── 子图标题 & 纵轴标签 ──────────────────────────────────────────────────────
VIOLIN_TITLES = [
    "(a) Posting Hour Entropy",
    "(b) Posting Interval (CV)",
    "(c) Burstiness Index",
]
VIOLIN_YLABELS = [
    "Shannon Entropy (bits)",
    "Coefficient of Variation",
    "Burstiness Index",
]
HOUR_DIST_TITLE  = "(d) Posting Hour Distribution"
HOUR_DIST_YLABEL = ""

GROUPS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]   # 用作字典 key / index
COLORS = ["limegreen", "tomato", "orange", "goldenrod"]
LABELS        = ["Human", "Bot", "Agent\n(Weibo)", "Agent\n(Moltbook)"]  # x轴 tick 显示
LEGEND_LABELS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]        # 图例显示


# ── 特征计算 ──────────────────────────────────────────────────────────────────
def _shannon_entropy(hour_counts: np.ndarray) -> float:
    """Shannon 熵（bits），输入为 24-bin 计数数组。"""
    total = hour_counts.sum()
    if total == 0:
        return np.nan
    p = hour_counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _interval_diffs(timestamps: list[datetime]) -> np.ndarray:
    """从排序后的时间戳列表提取非负时间间隔（秒）。"""
    ts_sorted = sorted(t for t in timestamps if t is not None)
    if len(ts_sorted) < 2:
        return np.array([], dtype=float)
    diffs = np.array([
        (ts_sorted[i+1] - ts_sorted[i]).total_seconds()
        for i in range(len(ts_sorted) - 1)
    ], dtype=float)
    return diffs[diffs > 0]


def _interval_cv(diffs: np.ndarray) -> float | None:
    """计算时间间隔的变异系数 CV = std / mean。"""
    if len(diffs) == 0 or diffs.mean() == 0:
        return None
    return float(diffs.std() / diffs.mean())


def _burstiness_index(diffs: np.ndarray) -> float | None:
    """Goh & Barabási 爆发性指数 B = (σ - μ) / (σ + μ)。
    B → -1: 完全机械式规律（bot 特征）
    B =  0: 随机 Poisson 过程
    B → +1: 高度爆发（人类特征）
    """
    if len(diffs) == 0:
        return None
    mu = diffs.mean()
    sigma = diffs.std()
    if sigma + mu == 0:
        return None
    return float((sigma - mu) / (sigma + mu))


def compute_temporal(timestamps: list[datetime]) -> dict:
    valid_ts = [t for t in timestamps if t is not None]
    n = len(valid_ts)

    # 小时熵
    if n >= 3:
        hours = np.array([t.hour for t in valid_ts])
        counts = np.bincount(hours, minlength=24).astype(float)
        h_ent = _shannon_entropy(counts)
    else:
        h_ent = np.nan

    # 间隔特征：CV 和 burstiness 复用同一组 diffs
    cv = np.nan
    burstiness = np.nan
    diffs = _interval_diffs(valid_ts)
    if len(diffs) > 0 and diffs.mean() > 0:
        cv = _interval_cv(diffs)
        cv = cv if cv is not None else np.nan
        b = _burstiness_index(diffs)
        burstiness = b if b is not None else np.nan

    return {"hour_entropy": h_ent, "interval_cv": cv, "burstiness": burstiness}


# ── 绘图工具 ──────────────────────────────────────────────────────────────────
def _violin(ax, data_groups: list[np.ndarray], group_keys: list[str],
            title: str, ylabel: str):
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
                  colors="black", linewidths=1.1, zorder=5) #linestyles="dashed"

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[i] for i in valid_keys],
                       rotation=0, ha="center", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=YTICK_LABEL_SIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=6)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def _adjusted_hour(h: int) -> float:
    """将小时重映射：凌晨4点→0（图底），凌晨3点→23（图顶）。"""
    return float((h - 4) % 24)


def _hour_distribution(ax, hour_groups: dict[str, list[float]], title: str, ylabel: str = "Time of Day"):
    matrix = []
    for group in GROUPS:
        arr = np.array(hour_groups[group], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            matrix.append(np.zeros(24, dtype=float))
            continue
        bins = np.bincount(arr.astype(int), minlength=24).astype(float)
        matrix.append(bins / bins.sum())

    mat = np.array(matrix, dtype=float).T
    im = ax.imshow(
        mat,
        origin="upper",
        aspect="auto",
        cmap="YlOrBr",
        extent=(-0.5, len(GROUPS) - 0.5, 24, 0),
    )

    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels(LABELS, fontsize=TICK_LABEL_SIZE)

    tick_vals = [0, 4, 8, 12, 16, 20, 24]
    tick_labels = ["04:00", "08:00", "12:00", "16:00", "20:00", "24:00", "Nextday\n04:00"]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels, fontsize=YTICK_LABEL_SIZE)

    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=6)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(24, 0)
    ax.spines[["top", "right"]].set_visible(False)
    return im


def plot_temporal(groups_weibo: dict, agents_moltbook: list):
    feat: dict[str, list] = {g: [] for g in GROUPS}

    for g in ("Human", "Bot", "Agent(Weibo)"):
        for u in groups_weibo.get(g, []):
            ts_list = [t["ts"] for t in u.get("tweets", [])]
            r = compute_temporal(ts_list)
            feat[g].append(r)

    for u in agents_moltbook:
        ts_list = [p["ts"] for p in u.get("posts", [])]
        r = compute_temporal(ts_list)
        feat["Agent(Moltbook)"].append(r)

    def _arr(gkey, metric):
        return np.array([f[metric] for f in feat[gkey]], dtype=float)

    # 每条博文的调整后发博小时（凌晨4点→0，凌晨3点→23）
    hours: dict[str, list] = {g: [] for g in GROUPS}
    for g in ("Human", "Bot", "Agent(Weibo)"):
        for u in groups_weibo.get(g, []):
            for t in u.get("tweets", []):
                ts = t.get("ts")
                if ts is not None:
                    hours[g].append(_adjusted_hour(ts.hour))
    for u in agents_moltbook:
        for p in u.get("posts", []):
            ts = p.get("ts")
            if ts is not None:
                hours["Agent(Moltbook)"].append(_adjusted_hour(ts.hour))

    fig, axes = plt.subplots(1, 4, figsize=(20,6))
    fig.suptitle("Temporal Feature",
                 fontsize=SUPTITLE_SIZE, y=0.95, fontweight="bold")

    # col0: hour_entropy
    data = [_arr(g, "hour_entropy") for g in GROUPS]
    _violin(axes[0], data, GROUPS, VIOLIN_TITLES[0], VIOLIN_YLABELS[0])

    # col1: interval_cv
    data = [_arr(g, "interval_cv") for g in GROUPS]
    _violin(axes[1], data, GROUPS, VIOLIN_TITLES[1], VIOLIN_YLABELS[1])

    # col2: burstiness
    data = [_arr(g, "burstiness") for g in GROUPS]
    _violin(axes[2], data, GROUPS, VIOLIN_TITLES[2], VIOLIN_YLABELS[2])

    # col3: posting hour distribution
    im = _hour_distribution(axes[3], hours, HOUR_DIST_TITLE, HOUR_DIST_YLABEL)

    patches = [mpatches.Patch(color=COLORS[i], label=LEGEND_LABELS[i]) for i in range(len(GROUPS))]
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=AXIS_LABEL_SIZE)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=LEGEND_SIZE, frameon=False)
    out = PLOTS_DIR / "f-temporal.pdf"
    plt.savefig(out)
    print(f"\nSaved figure to: {out}")


if __name__ == "__main__":
    print("Loading data...")
    weibo = load_weibo_groups()
    molts = load_moltbook_agents()
    print("Plotting temporal features...")
    plot_temporal(weibo, molts)
