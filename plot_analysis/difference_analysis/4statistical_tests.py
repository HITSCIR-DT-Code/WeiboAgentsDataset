"""
python 4statistical_tests.py
==============================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_weibo_groups, load_moltbook_agents

# ── 路径 ────────────────────────────────────────────────────────────────────
CUR_DIR   = Path(__file__).parent
PLOTS_DIR = CUR_DIR / "plots"
CACHE_DIR = CUR_DIR / ".cache"
PLOTS_DIR.mkdir(exist_ok=True)

PPL_CACHE_FILE  = CACHE_DIR / "ppl_cache.json"
SENT_CACHE_FILE = CACHE_DIR / "sentiment_cache.json"

OUTPUT_FILE = PLOTS_DIR / "statistical_tests.tex"

GROUPS = ["Human", "Bot", "Agent(Weibo)", "Agent(Moltbook)"]
GROUP_SHORT = {"Human": "Hum", "Bot": "Bot",
               "Agent(Weibo)": "Ag(W)", "Agent(Moltbook)": "Ag(M)"}

REFERENCE_DATE_STR = "2026-04-01"


# ==========================================================================
#  Cliff's Delta
# ==========================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """
    计算 Cliff's Delta 及效应等级。
    δ = (#{x_i > y_j} - #{x_i < y_j}) / (n_x * n_y)
    等级: |δ| < 0.147 → negligible, < 0.33 → small, < 0.474 → medium, else → large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, "--"
    # 向量化计算
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = (more - less) / (nx * ny)
    # 效应等级
    abs_d = abs(delta)
    if abs_d < 0.147:
        level = "negligible"
    elif abs_d < 0.33:
        level = "small"
    elif abs_d < 0.474:
        level = "medium"
    else:
        level = "large"
    return float(delta), level


def kruskal_epsilon_squared(H: float, n: int, k: int) -> float:
    """
    Kruskal-Wallis 效应量 ε² = H / ((n² - 1) / (n + 1))
    即 ε² = H * (n + 1) / (n² - 1) = H / (n - 1)
    """
    if n <= 1:
        return np.nan
    return H / (n - 1)


# ==========================================================================
#  特征计算（复用逻辑）
# ==========================================================================

def _span_days(ts_list) -> float | None:
    valid = [t for t in ts_list if t is not None]
    if len(valid) < 2:
        return None
    delta = (max(valid) - min(valid)).total_seconds() / 86400
    return max(delta, 1.0)


def _char_ttr(texts: list[str]) -> float | None:
    joined = "".join(texts)
    total = len(joined)
    if total == 0:
        return None
    return len(set(joined)) / total


def _shannon_entropy(hour_counts: np.ndarray) -> float:
    total = hour_counts.sum()
    if total == 0:
        return np.nan
    p = hour_counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_weibo_features(user: dict) -> dict:
    """计算单个 Weibo 用户的全部指标。"""
    tweets = user.get("tweets", [])
    valid_ts = [t["ts"] for t in tweets if t.get("ts") is not None]
    n_ts = len(valid_ts)

    # ── Behavior ──
    span = _span_days(valid_ts)
    posts_per_day = len(tweets) / span if span else np.nan

    comment_counts = [t.get("comments_count", 0) for t in tweets]
    avg_comments = float(np.mean(comment_counts)) if comment_counts else np.nan

    follower_ratio = user["followers_count"] / (user["friends_count"] + 1)

    age_days = user.get("account_age_days") or None
    months = age_days / 30 if age_days else None
    engagement = user["interactions_count"] / months if months else np.nan

    # ── Content ──
    texts = [t["text"] for t in tweets if t.get("text")]
    avg_text_len = float(np.mean([len(t) for t in texts])) if texts else np.nan
    ttr = _char_ttr(texts) if texts else np.nan
    if ttr is None:
        ttr = np.nan

    # ── Temporal ──
    hour_entropy = np.nan
    if n_ts >= 3:
        hours = np.array([t.hour for t in valid_ts])
        counts = np.bincount(hours, minlength=24).astype(float)
        hour_entropy = _shannon_entropy(counts)

    interval_cv = np.nan
    burstiness = np.nan
    if n_ts >= 2:
        ts_sorted = sorted(valid_ts)
        diffs = np.array([
            (ts_sorted[i+1] - ts_sorted[i]).total_seconds()
            for i in range(len(ts_sorted) - 1)
        ], dtype=float)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0 and diffs.mean() > 0:
            interval_cv = float(diffs.std() / diffs.mean())
            mu, sigma = diffs.mean(), diffs.std()
            if sigma + mu > 0:
                burstiness = float((sigma - mu) / (sigma + mu))

    return {
        "posts_per_day": posts_per_day,
        "avg_comments_per_post": avg_comments,
        "follower_ratio": follower_ratio,
        "engagement_per_month": engagement,
        "avg_text_len": avg_text_len,
        "char_ttr": ttr,
        "hour_entropy": hour_entropy,
        "interval_cv": interval_cv,
        "burstiness": burstiness,
    }


def compute_moltbook_features(user: dict) -> dict:
    """计算单个 Moltbook 用户的全部指标。"""
    posts = user.get("posts", [])
    valid_ts = [p["ts"] for p in posts if p.get("ts") is not None]
    n_ts = len(valid_ts)

    # ── Behavior ──
    span = _span_days(valid_ts)
    posts_per_day = len(posts) / span if span else np.nan

    comment_counts = [p.get("comment_counts", 0) for p in posts]
    avg_comments = float(np.mean(comment_counts)) if comment_counts else np.nan

    follower_ratio = user["followers"] / (user["following"] + 1)

    age_days = user.get("account_age_days") or None
    months = age_days / 30 if age_days else None
    engagement = user["karma"] / months if months else np.nan

    # ── Content ──
    texts = [p["text"] for p in posts if p.get("text")]
    avg_text_len = float(np.mean([len(t) for t in texts])) if texts else np.nan
    ttr = _char_ttr(texts) if texts else np.nan
    if ttr is None:
        ttr = np.nan

    # ── Temporal ──
    hour_entropy = np.nan
    if n_ts >= 3:
        hours = np.array([t.hour for t in valid_ts])
        counts = np.bincount(hours, minlength=24).astype(float)
        hour_entropy = _shannon_entropy(counts)

    interval_cv = np.nan
    burstiness = np.nan
    if n_ts >= 2:
        ts_sorted = sorted(valid_ts)
        diffs = np.array([
            (ts_sorted[i+1] - ts_sorted[i]).total_seconds()
            for i in range(len(ts_sorted) - 1)
        ], dtype=float)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0 and diffs.mean() > 0:
            interval_cv = float(diffs.std() / diffs.mean())
            mu, sigma = diffs.mean(), diffs.std()
            if sigma + mu > 0:
                burstiness = float((sigma - mu) / (sigma + mu))

    return {
        "posts_per_day": posts_per_day,
        "avg_comments_per_post": avg_comments,
        "follower_ratio": follower_ratio,
        "engagement_per_month": engagement,
        "avg_text_len": avg_text_len,
        "char_ttr": ttr,
        "hour_entropy": hour_entropy,
        "interval_cv": interval_cv,
        "burstiness": burstiness,
    }


# ==========================================================================
#  缓存加载（PPL / Sentiment）
# ==========================================================================

def _load_cache_metric(cache_path: Path, metric_key: str) -> dict[str, np.ndarray]:
    """从缓存文件加载指标，返回 {group: array}。"""
    result = {g: np.array([], dtype=float) for g in GROUPS}
    if not cache_path.exists():
        print(f"  Warning: cache not found → {cache_path}")
        return result
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for g in GROUPS:
        if g in data:
            vals = [item[metric_key] for item in data[g]
                    if item.get(metric_key) is not None]
            result[g] = np.array(vals, dtype=float)
    return result


# ==========================================================================
#  主流程
# ==========================================================================

# 指标定义: (display_name, metric_key, category)
METRICS = [
    ("Posts/Day",          "posts_per_day",         "Behavior"),
    ("Avg Comments",       "avg_comments_per_post", "Behavior"),
    ("Follower Ratio",     "follower_ratio",        "Behavior"),
    ("Engagement/Month",   "engagement_per_month",  "Behavior"),
    ("Avg Text Length",    "avg_text_len",          "Content"),
    ("Char TTR",           "char_ttr",              "Content"),
    ("Avg Perplexity",     "avg_ppl",              "Content"),
    ("Avg Sentiment",      "avg_sentiment",         "Content"),
    ("Hour Entropy",       "hour_entropy",          "Temporal"),
    ("Interval CV",        "interval_cv",           "Temporal"),
    ("Burstiness",         "burstiness",            "Temporal"),
]


def _format_p(p: float) -> str:
    """格式化 p 值：<0.001 显示为 <.001，否则保留 3 位小数。"""
    if p < 0.001:
        return "$<$.001"
    return f".{p:.3f}"[1:]  # 去掉前导零


def _format_delta(delta: float, level: str) -> str:
    """格式化 Cliff's delta。"""
    if np.isnan(delta):
        return "--"
    # 用缩写标注等级
    level_mark = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}
    return f"{delta:+.2f}\\textsuperscript{{{level_mark[level]}}}"


def run_tests():
    print("Loading data...")
    weibo = load_weibo_groups(verbose=True)
    molts = load_moltbook_agents(verbose=True)

    # ── 计算用户级特征 ──
    print("\nComputing features...")
    feat_arrays: dict[str, dict[str, np.ndarray]] = {g: {} for g in GROUPS}

    # Weibo 三组
    for g in GROUPS[:3]:
        all_feats = [compute_weibo_features(u) for u in weibo.get(g, [])]
        for metric_key in [m[1] for m in METRICS if m[1] not in ("avg_ppl", "avg_sentiment")]:
            vals = np.array([f[metric_key] for f in all_feats], dtype=float)
            feat_arrays[g][metric_key] = vals[np.isfinite(vals)]

    # Moltbook
    all_feats_molt = [compute_moltbook_features(u) for u in molts]
    for metric_key in [m[1] for m in METRICS if m[1] not in ("avg_ppl", "avg_sentiment")]:
        vals = np.array([f[metric_key] for f in all_feats_molt], dtype=float)
        feat_arrays["Agent(Moltbook)"][metric_key] = vals[np.isfinite(vals)]

    # PPL / Sentiment from cache
    ppl_data = _load_cache_metric(PPL_CACHE_FILE, "avg_ppl")
    sent_data = _load_cache_metric(SENT_CACHE_FILE, "avg_sentiment")
    for g in GROUPS:
        feat_arrays[g]["avg_ppl"] = ppl_data[g][np.isfinite(ppl_data[g])]
        feat_arrays[g]["avg_sentiment"] = sent_data[g][np.isfinite(sent_data[g])]

    # ── 统计检验 ──
    print("\nRunning statistical tests...")
    results = []

    for display_name, metric_key, category in METRICS:
        arrays = [feat_arrays[g][metric_key] for g in GROUPS]
        # 过滤掉空数组
        valid_arrays = [a for a in arrays if len(a) >= 2]
        valid_groups = [g for g, a in zip(GROUPS, arrays) if len(a) >= 2]

        # Kruskal-Wallis
        if len(valid_arrays) >= 2:
            H_stat, p_kw = stats.kruskal(*valid_arrays)
            n_total = sum(len(a) for a in valid_arrays)
            eps_sq = kruskal_epsilon_squared(H_stat, n_total, len(valid_arrays))
        else:
            H_stat, p_kw, eps_sq = np.nan, np.nan, np.nan

        # 两两 Mann-Whitney U + Cliff's Delta
        pairs = list(combinations(range(len(GROUPS)), 2))
        n_pairs = len(pairs)
        pair_results = []

        for i, j in pairs:
            arr_i, arr_j = arrays[i], arrays[j]
            if len(arr_i) < 2 or len(arr_j) < 2:
                pair_results.append({
                    "pair": f"{GROUP_SHORT[GROUPS[i]]} vs {GROUP_SHORT[GROUPS[j]]}",
                    "U": np.nan, "p_adj": np.nan,
                    "delta": np.nan, "level": "--",
                })
                continue
            U_stat, p_mw = stats.mannwhitneyu(arr_i, arr_j, alternative="two-sided")
            p_adj = min(p_mw * n_pairs, 1.0)  # Bonferroni
            delta, level = cliffs_delta(arr_i, arr_j)
            pair_results.append({
                "pair": f"{GROUP_SHORT[GROUPS[i]]} vs {GROUP_SHORT[GROUPS[j]]}",
                "U": U_stat, "p_adj": p_adj,
                "delta": delta, "level": level,
            })

        results.append({
            "display_name": display_name,
            "category": category,
            "metric_key": metric_key,
            "H": H_stat, "p_kw": p_kw, "eps_sq": eps_sq,
            "n": [len(arrays[i]) for i in range(len(GROUPS))],
            "pairs": pair_results,
        })

        # 终端输出
        print(f"\n  {display_name} ({category})")
        print(f"    Kruskal-Wallis: H={H_stat:.2f}, p={p_kw:.2e}, "
              f"eps^2={eps_sq:.4f}")
        print(f"    Sample sizes: {dict(zip(GROUPS, [len(a) for a in arrays]))}")
        for pr in pair_results:
            if not np.isnan(pr["delta"]):
                print(f"    {pr['pair']:20s}: U={pr['U']:.0f}, "
                      f"p_adj={pr['p_adj']:.2e}, "
                      f"delta={pr['delta']:+.3f} ({pr['level']})")

    # ── 生成 LaTeX ──
    print("\n\nGenerating LaTeX table...")
    latex = _generate_latex(results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved to: {OUTPUT_FILE}")
    print("\n── LaTeX Preview ──")
    print(latex[:2000])


def _generate_latex(results: list[dict]) -> str:
    """生成两张 LaTeX 表格：Table 1 总览 + Table 2 两两比较。"""
    lines = []

    # ===== Table 1: Kruskal-Wallis 总览 =====
    lines.append(r"% Table 1: Kruskal-Wallis omnibus test results")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Kruskal-Wallis omnibus test results across four groups.}")
    lines.append(r"\label{tab:kruskal}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{Metric} & "
                 r"\textbf{$H$} & \textbf{$p$} & "
                 r"\textbf{$\varepsilon^2$} & \textbf{Sig.} \\")
    lines.append(r"\midrule")

    prev_cat = ""
    for r in results:
        cat_cell = ""
        if r["category"] != prev_cat:
            if prev_cat:
                lines.append(r"\midrule")
            cat_cell = r["category"]
            prev_cat = r["category"]

        sig = ""
        if not np.isnan(r["p_kw"]):
            if r["p_kw"] < 0.001:
                sig = "***"
            elif r["p_kw"] < 0.01:
                sig = "**"
            elif r["p_kw"] < 0.05:
                sig = "*"

        p_str = "$<$.001" if r["p_kw"] < 0.001 else f"{r['p_kw']:.3f}"
        eps_str = f"{r['eps_sq']:.3f}" if not np.isnan(r["eps_sq"]) else "--"

        lines.append(
            f"  {cat_cell} & {r['display_name']} & "
            f"{r['H']:.1f} & {p_str} & {eps_str} & {sig} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\parbox{\linewidth}{\footnotesize "
                 r"\textit{Note.} $\varepsilon^2$ = rank-based epsilon squared "
                 r"(effect size for Kruskal-Wallis). "
                 r"Significance: *** $p<.001$, ** $p<.01$, * $p<.05$.}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append("")

    # ===== Table 2: Pairwise Mann-Whitney + Cliff's Delta =====
    pair_names = results[0]["pairs"]
    n_pairs = len(pair_names)

    lines.append(r"% Table 2: Pairwise Mann-Whitney U tests with Cliff's Delta")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise Mann-Whitney $U$ tests with Bonferroni correction "
                 r"and Cliff's $\delta$ effect sizes.}")
    lines.append(r"\label{tab:pairwise}")
    lines.append(r"\small")

    # 列: Metric | pair1_p | pair1_δ | pair2_p | pair2_δ | ...
    col_spec = "l" + "rl" * n_pairs
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: pair names
    header_parts = [r"\textbf{Metric}"]
    for pr in pair_names:
        header_parts.append(r"\multicolumn{2}{c}{\textbf{" + pr["pair"] + "}}")
    lines.append(" & ".join(header_parts) + r" \\")

    # Header row 2: p_adj / δ
    sub_header = [""]
    for _ in pair_names:
        sub_header.append(r"$p_{\text{adj}}$")
        sub_header.append(r"$\delta$")
    lines.append(" & ".join(sub_header) + r" \\")
    lines.append(r"\midrule")

    prev_cat = ""
    for r in results:
        if r["category"] != prev_cat:
            if prev_cat:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(1 + 2*n_pairs) +
                         r"}{l}{\textit{" + r["category"] + r"}} \\")
            prev_cat = r["category"]

        row_parts = [f"~~{r['display_name']}"]
        for pr in r["pairs"]:
            if np.isnan(pr["p_adj"]):
                row_parts.append("--")
                row_parts.append("--")
            else:
                p_str = "$<$.001" if pr["p_adj"] < 0.001 else f"{pr['p_adj']:.3f}"
                d_str = _format_delta(pr["delta"], pr["level"])
                row_parts.append(p_str)
                row_parts.append(d_str)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\parbox{\linewidth}{\footnotesize "
                 r"\textit{Note.} $p_{\text{adj}}$: Bonferroni-corrected $p$-values "
                 r"from Mann-Whitney $U$ tests. "
                 r"$\delta$: Cliff's Delta with effect size level "
                 r"(N=negligible, S=small, M=medium, L=large). "
                 r"Positive $\delta$ indicates the first group has larger values.}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_tests()
