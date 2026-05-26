"""
python 1behavior_content_analysis.py
=====================================
Combined 2×4 figure: Behavior (top row) + Content (bottom row)
→ difference_analysis/plots/behavior_content.png

"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# ── 路径 ────────────────────────────────────────────────────────────────────
CUR_DIR  = Path(__file__).parent          # difference_analysis/
PAR_DIR  = CUR_DIR.parent                  # plot_analysis/
WEIBO_DIR   = PAR_DIR / "dataset" / "Weibo"
MOLT_DIR    = PAR_DIR / "dataset" / "Moltbook"
PLOTS_DIR   = CUR_DIR / "plots"
CACHE_DIR   = CUR_DIR / ".cache"

PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── 文件路径 ─────────────────────────────────────────────────────────────────
LABELS_FILE       = WEIBO_DIR / "Weibo_Labels.json"
PROFILE_FILE      = WEIBO_DIR / "user_profile.jsonl"
TWEETS_FILE       = WEIBO_DIR / "user_tweets.jsonl"
MOLT_INFO_FILE    = MOLT_DIR  / "user_info.jsonl"
MOLT_POSTS_FILE   = MOLT_DIR  / "user_history_posts.jsonl"
PPL_CACHE_FILE    = CACHE_DIR / "ppl_cache.json"
SENT_CACHE_FILE   = CACHE_DIR / "sentiment_cache.json"

REFERENCE_DATE = datetime(2026, 4, 1)

# ── 颜色 & 标签 ──────────────────────────────────────────────────────────────
GROUPS       = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]
COLORS       = ["limegreen", "tomato", "orange", "darkorange"]
XTICK_LABELS = ["Human", "Bot", "Agent\nWeibo", "Agent\nMoltbook"]
LEGEND_LABELS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]

# ── 统一字体大小 ─────────────────────────────────────────────────────────────
ROW_TITLE_SIZE     = 24
SUBPLOT_TITLE_SIZE = 20
AXIS_LABEL_SIZE    = 16
TICK_LABEL_SIZE    = 16
YTICK_LABEL_SIZE   = 14


# ==========================================================================
#  数据加载（内联自 data_loader.py）
# ==========================================================================

def _age_days(date_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> float | None:
    """返回账号年龄（天数），解析失败返回 None。"""
    for f in (fmt, "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), f)
            delta = (REFERENCE_DATE - dt).total_seconds() / 86400
            return max(delta, 1.0)
        except ValueError:
            continue
    return None


def _parse_ts(date_str: str) -> datetime | None:
    """解析推文时间戳，失败返回 None。"""
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), f)
        except ValueError:
            continue
    return None


def load_weibo_groups(verbose: bool = True) -> dict[str, list]:
    """加载微博数据，返回 {"Human": [...], "Bot": [...], "Agent(Weibo)": [...]}。"""
    # 1. 标签
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels: dict[str, int] = json.load(f)

    label_map: dict[str, str] = {}
    for uid, lbl in labels.items():
        if lbl == 0:
            label_map[uid] = "Human"
        elif lbl == 2:
            label_map[uid] = "Bot"
        elif lbl == -1:
            label_map[uid] = "Agent(Weibo)"

    # 2. Profile
    profiles: dict[str, dict] = {}
    with open(PROFILE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(rec.get("uid", ""))
            if uid not in label_map:
                continue
            age = _age_days(rec.get("created_at", ""))
            profiles[uid] = {
                "uid": uid,
                "label_group": label_map[uid],
                "followers_count":    rec.get("followers_count", 0) or 0,
                "friends_count":      rec.get("friends_count", 0) or 0,
                "statuses_count":     rec.get("statuses_count", 0) or 0,
                "interactions_count": rec.get("interactions_count", 0) or 0,
                "verified":              bool(rec.get("verified", False)),
                "default_profile_image": bool(rec.get("default_profile_image", True)),
                "description": rec.get("description", "") or "",
                "account_age_days": age,
                "tweets": [],
            }

    # 3. 推文聚合
    tweet_buckets: dict[str, list] = defaultdict(list)
    with open(TWEETS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(rec.get("uid", ""))
            if uid not in profiles:
                continue
            ts = _parse_ts(rec.get("created_at", ""))
            tweet_buckets[uid].append({
                "text":           rec.get("text", "") or "",
                "ts":             ts,
                "is_retweet":     bool(rec.get("is_retweet", False)),
                "has_image":      bool(rec.get("images")),
                "comments_count": rec.get("comments_count", 0) or 0,
            })

    for uid, twts in tweet_buckets.items():
        if uid in profiles:
            profiles[uid]["tweets"] = sorted(
                twts, key=lambda x: x["ts"] or datetime.min
            )

    # 4. 分组
    groups: dict[str, list] = {"Human": [], "Bot": [], "Agent(Weibo)": []}
    for uid, pdata in profiles.items():
        g = pdata["label_group"]
        if g in groups:
            groups[g].append(pdata)

    if verbose:
        for g, lst in groups.items():
            print(f"  Weibo {g:14s}: {len(lst)} users")

    return groups


def load_moltbook_agents(verbose: bool = True) -> list:
    """加载 Moltbook 数据，返回 [user_dict, ...]。"""
    # 1. 用户信息
    users: dict[str, dict] = {}
    with open(MOLT_INFO_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = rec.get("author", "")
            if not author:
                continue
            age = _age_days(rec.get("join_time", ""), fmt="%Y-%m-%d")
            users[author] = {
                "author":      author,
                "followers":   rec.get("followers", 0) or 0,
                "following":   rec.get("following", 0) or 0,
                "karma":       rec.get("karma", 0) or 0,
                "verified":    bool(rec.get("verified", False)),
                "description": rec.get("discription", "") or "",
                "account_age_days": age,
                "posts": [],
            }

    # 2. 帖子聚合
    with open(MOLT_POSTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = rec.get("author", "")
            if author not in users:
                continue
            ts = _parse_ts(rec.get("time", ""))
            text = rec.get("short_content", "") or rec.get("title", "") or ""
            users[author]["posts"].append({
                "text":           text,
                "ts":             ts,
                "comment_counts": rec.get("comment_counts", 0) or 0,
                "vote_count":     rec.get("vote_count", 0) or 0,
            })

    for u in users.values():
        u["posts"] = sorted(u["posts"], key=lambda x: x["ts"] or datetime.min)

    result = list(users.values())
    if verbose:
        print(f"  Moltbook agent: {len(result)} users")
    return result


# ==========================================================================
#  特征计算
# ==========================================================================

def _span_days(ts_list: list) -> float | None:
    """首末时间戳之差（天），不足 2 条有效时间戳时返回 None。"""
    valid = [t for t in ts_list if t is not None]
    if len(valid) < 2:
        return None
    delta = (max(valid) - min(valid)).total_seconds() / 86400
    return max(delta, 1.0)


def _char_ttr(texts: list[str]) -> float | None:
    """字符级 type-token ratio：unique chars / total chars（拼接所有文本）。"""
    joined = "".join(texts)
    total = len(joined)
    if total == 0:
        return None
    return len(set(joined)) / total


# ── Behavior 特征 ──

def compute_weibo_behavior(user: dict) -> dict | None:
    tweets = user.get("tweets", [])
    span = _span_days([t["ts"] for t in tweets])
    posts_per_day = len(tweets) / span if span else None
    age_days = user.get("account_age_days") or None
    months = age_days / 30 if age_days else None
    comment_counts = [t.get("comments_count", 0) for t in tweets]
    if len(comment_counts) >= 1:
        avg_comm = np.mean(comment_counts)
    else:
        avg_comm = None
    return {
        "posts_per_day":         posts_per_day,
        "follower_ratio":        user["followers_count"] / (user["friends_count"] + 1),
        "engagement_per_month":  user["interactions_count"] / months if months else None,
        "avg_comments_per_post": avg_comm,
    }


def compute_moltbook_behavior(user: dict) -> dict | None:
    posts = user.get("posts", [])
    span = _span_days([p["ts"] for p in posts])
    posts_per_day = len(posts) / span if span else None
    age_days = user.get("account_age_days") or None
    months = age_days / 30 if age_days else None
    comment_counts = [p.get("comment_counts", 0) for p in posts]
    if len(comment_counts) >= 1:
        avg_comm = np.mean(comment_counts)
    else:
        avg_comm = None
    return {
        "posts_per_day":         posts_per_day,
        "follower_ratio":        user["followers"] / (user["following"] + 1),
        "engagement_per_month":  user["karma"] / months if months else None,
        "avg_comments_per_post": avg_comm,
    }


# ── Content 特征 ──

def compute_weibo_content(user: dict) -> dict | None:
    tweets = user.get("tweets", [])
    if not tweets:
        return None
    texts = [t["text"] for t in tweets if t["text"]]
    if not texts:
        return None
    avg_len = np.mean([len(t) for t in texts])
    ttr = _char_ttr(texts)
    return {
        "avg_text_len": avg_len,
        "char_ttr":     ttr,
    }


def compute_moltbook_content(user: dict) -> dict | None:
    posts = user.get("posts", [])
    if not posts:
        return None
    texts = [p["text"] for p in posts if p["text"]]
    if not texts:
        return None
    avg_len = np.mean([len(t) for t in texts])
    ttr = _char_ttr(texts)
    return {
        "avg_text_len": avg_len,
        "char_ttr":     ttr,
    }


# ==========================================================================
#  缓存加载
# ==========================================================================

def _load_json_cache(path: Path) -> dict | None:
    """加载 JSON 缓存文件，不存在则返回 None。"""
    if not path.exists():
        print(f"Warning: cache file not found → {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Cache loaded → {path}  ({sum(len(v) for v in data.values())} users)")
    return data


# ==========================================================================
#  统一 violin 绘图函数（无 n=xxx 标注）
# ==========================================================================

def _violin(ax, data_groups: list[np.ndarray], group_keys: list[str],
            title: str, ylabel: str, log_scale: bool = False):
    """在 ax 上绘制并排 violin，叠加中位数横线。不显示样本量标注。"""
    valid_data   = []
    valid_colors = []
    valid_keys   = []
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

    # 中位数横线
    for pos, arr in zip(positions, valid_data):
        med = np.median(arr)
        ax.hlines(med, pos - 0.15, pos + 0.15,
                  colors="black", linewidths=1.1, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([XTICK_LABELS[i] for i in valid_keys],
                       rotation=0, ha="center", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=YTICK_LABEL_SIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=6)
    lbl = ylabel + " (log)" if log_scale else ylabel
    ax.set_ylabel(lbl, fontsize=AXIS_LABEL_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


# ==========================================================================
#  合并绘图
# ==========================================================================

# 上排：行为指标
BEHAVIOR_METRICS = [
    ("posts_per_day",         "Posts per Day",             True),
    ("avg_comments_per_post", "Avg Comments per Post",     True),
    ("follower_ratio",        "Follower / Following",      True),
    ("engagement_per_month",  "Engagement per Month",      True),
]

# 下排：内容指标
CONTENT_METRICS = [
    ("avg_text_len",   "Avg Text Length",              True),
    ("char_ttr",       "Char-level TTR",               False),
    ("avg_ppl",        "Avg Perplexity",               False),   # 来自缓存
    ("avg_sentiment",  "Avg Sentiment Score",          False),   # 来自缓存
]


def plot_combined(groups_weibo: dict, agents_moltbook: list):
    # ── 加载缓存 ──
    ppl_cache       = _load_json_cache(PPL_CACHE_FILE)
    sentiment_cache = _load_json_cache(SENT_CACHE_FILE)

    # ── 计算 Behavior 特征 ──
    feat_behavior: dict[str, list] = {g: [] for g in GROUPS}
    for g in GROUPS[:-1]:
        for u in groups_weibo.get(g, []):
            r = compute_weibo_behavior(u)
            if r:
                feat_behavior[g].append(r)
    for u in agents_moltbook:
        r = compute_moltbook_behavior(u)
        if r:
            feat_behavior[GROUPS[-1]].append(r)

    # ── 计算 Content 特征 ──
    feat_content: dict[str, list] = {g: [] for g in GROUPS}
    for g in GROUPS[:-1]:
        for u in groups_weibo.get(g, []):
            r = compute_weibo_content(u)
            if r:
                feat_content[g].append(r)
    for u in agents_moltbook:
        r = compute_moltbook_content(u)
        if r:
            feat_content[GROUPS[-1]].append(r)

    def _arr(feat_dict, group_key, metric):
        return np.array(
            [f[metric] for f in feat_dict[group_key] if f.get(metric) is not None],
            dtype=float
        )

    def _arr_cache(cache, group_key, metric):
        """从缓存 dict 中提取指标数组。"""
        if cache is None or group_key not in cache:
            return np.array([], dtype=float)
        return np.array(
            [f[metric] for f in cache[group_key] if f.get(metric) is not None],
            dtype=float
        )

    # ── 2×4 布局 ──
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 10))

    # ── 上排：Behavior ──
    behavior_ylabels   = ["Num", "Num", "Ratio", "Num"]
    behavior_subtitles = ["(a) Daily Posts", "(b) Average Comments per Post",
                          "(c) Follower / Following", "(d) Monthly Engagement"]
    for i, (ax, (key, _, use_log), ylabel, subtitle) in enumerate(zip(
            axes[0], BEHAVIOR_METRICS, behavior_ylabels, behavior_subtitles)):
        gs = GROUPS[:3] if key == "engagement_per_month" else GROUPS
        data = [np.log1p(_arr(feat_behavior, g, key)) if use_log
                else _arr(feat_behavior, g, key)
                for g in gs]
        _violin(ax, data, gs, subtitle, ylabel, log_scale=use_log)

    # ── 下排：Content ──
    # col 1: avg_text_len (log1p)
    data = [np.log1p(_arr(feat_content, g, "avg_text_len")) for g in GROUPS]
    _violin(axes[1, 0], data, GROUPS,
            "(e) Average Length", "Num", log_scale=True)

    # col 2: char_ttr
    data = [_arr(feat_content, g, "char_ttr") for g in GROUPS]
    _violin(axes[1, 1], data, GROUPS,
            "(f) Text (Char) Diversity", "TTR")

    # col 3: avg_ppl (from ppl cache)
    data = [_arr_cache(ppl_cache, g, "avg_ppl") for g in GROUPS]
    _violin(axes[1, 2], data, GROUPS,
            "(g) Perplexity", "")

    # col 4: avg_sentiment (from sentiment cache)
    data = [_arr_cache(sentiment_cache, g, "avg_sentiment") for g in GROUPS]
    _violin(axes[1, 3], data, GROUPS,
            "(h) Sentiment", "")

    # 情感分数 y 轴刻度换成英文情感标签
    ax_sent = axes[1, 3]
    ymin_s, ymax_s = ax_sent.get_ylim()
    ax_sent.set_ylim(min(ymin_s, -2.2), max(ymax_s, 2.2))
    ax_sent.set_yticks([-2, -1, 0, 1, 2])
    ax_sent.set_yticklabels(
        ["Very\nNegative", "Negative", "Neutral", "Positive", "Very\nPositive"],
        ha="right", fontsize=YTICK_LABEL_SIZE)

    # ── 排标题（居中置于每排上方）──
    # tight_layout 后 axes 的 position 已更新，直接取图坐标
    # rect 顶部留白 (top=0.90) + h_pad 增加行间距
    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=6.0)

    pos0 = axes[0, 1].get_position()  # 上排中间子图的 bbox
    pos1 = axes[1, 1].get_position()  # 下排中间子图的 bbox
    gap = 0.05  # 标题与子图顶部间距（图坐标）

    fig.text(0.5, pos0.y1 + gap, "Interaction Features",
             ha="center", va="bottom", fontsize=ROW_TITLE_SIZE,
             fontweight="bold", transform=fig.transFigure)

    fig.text(0.5, pos1.y1 + gap, "Content Features",
             ha="center", va="bottom", fontsize=ROW_TITLE_SIZE,
             fontweight="bold", transform=fig.transFigure)

    # ── 图例 ──
    patches = [mpatches.Patch(color=COLORS[i], label=LEGEND_LABELS[i])
               for i in range(len(GROUPS))]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False)

    out = PLOTS_DIR / "f-behavior_content.pdf"
    plt.savefig(out)
    print(f"\nSaved figure to: {out}")


# ==========================================================================
#  入口
# ==========================================================================

if __name__ == "__main__":
    print("Loading data...")
    weibo = load_weibo_groups()
    molts = load_moltbook_agents()
    print("Plotting combined behavior + content features...")
    plot_combined(weibo, molts)
