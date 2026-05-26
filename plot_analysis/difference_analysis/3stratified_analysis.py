"""
python 3stratified_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_weibo_groups

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = PLOTS_DIR / "stratified_table.tex"

# ── 分层边界 ─────────────────────────────────────────────────────────────────
STRATA = [
    ("<1 yr",   0,    365),
    ("1--3 yr", 365,  1095),
    (">3 yr",   1095, 1e9),
]

GROUPS = ["Human", "Bot", "Agent(Weibo)"]
GROUP_DISPLAY = {"Human": "Human", "Bot": "Bot", "Agent(Weibo)": "Agent"}

# ── 指标计算函数 ──────────────────────────────────────────────────────────────

def _shannon_entropy(hour_counts: np.ndarray) -> float:
    total = hour_counts.sum()
    if total == 0:
        return np.nan
    p = hour_counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _char_ttr(texts: list[str]) -> float | None:
    joined = "".join(texts)
    total = len(joined)
    if total == 0:
        return None
    return len(set(joined)) / total


def compute_user_features(user: dict) -> dict:
    """计算单个 Weibo 用户的全部关键指标。"""
    tweets = user.get("tweets", [])
    valid_ts = [t["ts"] for t in tweets if t.get("ts") is not None]
    n_ts = len(valid_ts)

    # ── Hour Entropy ──
    hour_entropy = np.nan
    if n_ts >= 3:
        hours = np.array([t.hour for t in valid_ts])
        counts = np.bincount(hours, minlength=24).astype(float)
        hour_entropy = _shannon_entropy(counts)

    # ── Interval CV ──
    interval_cv = np.nan
    if n_ts >= 2:
        ts_sorted = sorted(valid_ts)
        diffs = np.array([
            (ts_sorted[i+1] - ts_sorted[i]).total_seconds()
            for i in range(len(ts_sorted) - 1)
        ], dtype=float)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0 and diffs.mean() > 0:
            interval_cv = float(diffs.std() / diffs.mean())

    # ── Burstiness ──
    burstiness = np.nan
    if n_ts >= 2:
        ts_sorted = sorted(valid_ts)
        diffs = np.array([
            (ts_sorted[i+1] - ts_sorted[i]).total_seconds()
            for i in range(len(ts_sorted) - 1)
        ], dtype=float)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            mu = diffs.mean()
            sigma = diffs.std()
            if sigma + mu > 0:
                burstiness = float((sigma - mu) / (sigma + mu))

    # ── Char TTR ──
    texts = [t["text"] for t in tweets if t.get("text")]
    ttr = _char_ttr(texts) if texts else np.nan
    if ttr is None:
        ttr = np.nan

    # ── Posts per Day ──
    age_days = user.get("account_age_days")
    posts_per_day = np.nan
    if age_days and age_days > 0 and n_ts >= 1:
        # 使用发博时间跨度
        ts_sorted = sorted(valid_ts)
        span = (ts_sorted[-1] - ts_sorted[0]).total_seconds() / 86400
        if span >= 1:
            posts_per_day = n_ts / span

    return {
        "hour_entropy": hour_entropy,
        "interval_cv":  interval_cv,
        "burstiness":   burstiness,
        "char_ttr":     ttr,
        "posts_per_day": posts_per_day,
    }


# ── 分层统计 ──────────────────────────────────────────────────────────────────

def _median_iqr(values: np.ndarray) -> str:
    """返回格式化字符串: median (IQR_low–IQR_high)。如果数据不足则返回 '--'。"""
    arr = values[np.isfinite(values)]
    if len(arr) < 3:
        return "--"
    med = np.median(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    return f"{med:.2f} ({q1:.2f}--{q3:.2f})"


def _median_only(values: np.ndarray) -> str:
    """仅返回中位数。"""
    arr = values[np.isfinite(values)]
    if len(arr) < 3:
        return "--"
    return f"{np.median(arr):.3f}"


def run_stratified_analysis():
    print("Loading Weibo data...")
    groups = load_weibo_groups(verbose=True)

    # 计算每个用户的特征
    user_features: dict[str, list[tuple[float, dict]]] = {g: [] for g in GROUPS}
    for g in GROUPS:
        for u in groups.get(g, []):
            age = u.get("account_age_days")
            if age is None:
                continue
            feats = compute_user_features(u)
            user_features[g].append((age, feats))

    # 构建表格数据
    metrics = [
        ("Hour Ent.", "hour_entropy"),
        ("Intv. CV",  "interval_cv"),
        ("Burst.",    "burstiness"),
        ("Char TTR",  "char_ttr"),
        ("Posts/Day", "posts_per_day"),
    ]

    print("\n" + "="*80)
    print("Stratified Analysis Results (Weibo only, median values)")
    print("="*80)

    # 用于 LaTeX 表格
    latex_lines = []
    latex_lines.append(r"\begin{tabular}{ll" + "c" * len(GROUPS) + "}")
    latex_lines.append(r"\toprule")
    header = r"\textbf{Age Stratum} & \textbf{Metric} & " + \
             " & ".join([rf"\textbf{{{GROUP_DISPLAY[g]}}}" for g in GROUPS]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    for si, (stratum_name, lo, hi) in enumerate(STRATA):
        print(f"\n── Stratum: {stratum_name} (age in [{lo}, {hi}) days) ──")

        # 各组在此层的用户数
        counts = {}
        for g in GROUPS:
            users_in_stratum = [(age, f) for (age, f) in user_features[g]
                                if lo <= age < hi]
            counts[g] = len(users_in_stratum)
            print(f"  {g:14s}: n={counts[g]}")

        first_in_stratum = True
        for metric_name, metric_key in metrics:
            row_values = []
            for g in GROUPS:
                users_in_stratum = [(age, f) for (age, f) in user_features[g]
                                    if lo <= age < hi]
                vals = np.array([f[metric_key] for (_, f) in users_in_stratum],
                                dtype=float)
                row_values.append(_median_only(vals))

            # 打印到终端
            print(f"    {metric_name:12s}: " +
                  "  ".join([f"{g[:6]:>7s}={v}" for g, v in zip(GROUPS, row_values)]))

            # LaTeX 行
            if first_in_stratum:
                stratum_cell = rf"\multirow{{{len(metrics)}}}{{*}}{{{stratum_name}}}"
                first_in_stratum = False
            else:
                stratum_cell = ""

            latex_row = f"  {stratum_cell} & {metric_name} & " + \
                        " & ".join(row_values) + r" \\"
            latex_lines.append(latex_row)

        # 添加 n= 行
        n_row = f"  & \\textit{{n}} & " + \
                " & ".join([str(counts[g]) for g in GROUPS]) + r" \\"
        latex_lines.append(n_row)

        if si < len(STRATA) - 1:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    # 写出 LaTeX 文件
    latex_content = "\n".join(latex_lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"\n\nLaTeX table saved to: {OUTPUT_FILE}")
    print("\n── Generated LaTeX ──")
    print(latex_content)


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_stratified_analysis()
