"""
  cd baselines-moltbook/SGBot
  python preprocess.py
"""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Moltbook"
PROC_DIR = BASE_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "listed_count_proxy_interactions_count",
    "has_custom_profile_image",
    "verified",
    "account_age_days",
    "statuses_per_day",
    "followers_per_day",
    "friends_per_day",
    "listed_per_day_proxy_interactions_per_day",
    "follower_friend_ratio",
    "screen_name_length",
    "screen_name_digits",
    "name_length_proxy_screen_name_length",
    "name_digits_proxy_screen_name_digits",
    "description_length",
    "screen_name_bigram_likelihood",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Moltbook data for SGBot OOD")
    parser.add_argument(
        "--reference_date", type=str, default="2026-04-01",
        help="账号年龄截止日期，格式 YYYY-MM-DD（默认 2026-04-01）",
    )
    parser.add_argument(
        "--weibo_processed_dir",
        type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "SGBot" / "processed"),
        help="Weibo SGBot processed 目录，用于读取 bigram_stats.json（默认自动推断）",
    )
    return parser.parse_args()


# ── I/O 工具 ──────────────────────────────────────────────────────────────────

def iterate_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def count_digits(text: str) -> int:
    return sum(1 for ch in text if ch.isdigit())


# ── 日期解析 ──────────────────────────────────────────────────────────────────

def parse_join_time(value: str):
    """
    Moltbook 的 join_time 格式为 YYYY-MM-DD（仅日期，无时分秒）。
    与 Weibo 的 created_at（YYYY-MM-DD HH:MM:SS）不同，需单独解析。
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def compute_account_age_days(join_time_str: str, reference_time: datetime) -> float:
    """从 join_time 到 reference_date 的天数，最小为 1.0。"""
    dt = parse_join_time(join_time_str)
    if dt is None:
        return 1.0
    days = (reference_time - dt).days
    return max(float(days), 1.0)


# ── bigram 计算 ───────────────────────────────────────────────────────────────

def compute_bigram_likelihood(screen_name: str, bigram_stats: dict) -> float:
    """
    使用来自 Weibo 训练集的 bigram_stats 计算 screen_name 的 bigram likelihood。
    与 Weibo SGBot preprocess.py 中的 compute_bigram_likelihood 逻辑完全一致。
    """
    normalized = normalize_text(screen_name)
    if len(normalized) < 2:
        return 0.0

    default_probability = bigram_stats["default_probability"]
    probabilities = bigram_stats["probabilities"]
    log_sum = 0.0
    count = 0

    for i in range(len(normalized) - 1):
        bigram = normalized[i: i + 2]
        probability = probabilities.get(bigram, default_probability)
        log_sum += math.log(probability)
        count += 1

    return float(math.exp(log_sum / count)) if count > 0 else 0.0


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_user_info() -> dict:
    """返回 {author: info_dict}"""
    users = {}
    for item in iterate_jsonl(DATA_DIR / "user_info.jsonl"):
        author = item.get("author", "")
        if author:
            users[author] = item
    return users


def load_statuses_count(known_authors: set) -> dict:
    """
    流式遍历 user_history_posts.jsonl，统计每用户帖子行数作为 statuses_count 近似。
    返回 {author: count}
    """
    counts: dict = defaultdict(int)
    for item in tqdm(
        iterate_jsonl(DATA_DIR / "user_history_posts.jsonl"),
        desc="  Counting history posts",
        mininterval=5.0,
    ):
        author = item.get("author", "")
        if author in known_authors:
            counts[author] += 1
    return dict(counts)


# ── 特征提取 ──────────────────────────────────────────────────────────────────

def extract_feature_row(
    author: str,
    info: dict,
    statuses_count: int,
    reference_time: datetime,
    bigram_stats: dict,
) -> list:
    """
    提取单个用户的 18 维 SGBot 特征。
    特征顺序与 Weibo SGBot FEATURE_NAMES 完全一致。
    """
    followers_count = float(safe_int(info.get("followers")))
    friends_count   = float(safe_int(info.get("following")))
    karma           = float(safe_int(info.get("karma")))
    verified        = float(safe_bool(info.get("verified")))
    description     = normalize_text(info.get("discription"))  # 注意 Moltbook 字段拼写
    join_time       = info.get("join_time", "")

    # 发博数（来自 user_history_posts 统计）
    sc = float(statuses_count)

    # 账号年龄（Moltbook join_time 仅有日期精度）
    account_age_days = compute_account_age_days(join_time, reference_time)

    # 比率类特征
    statuses_per_day   = sc / account_age_days
    followers_per_day  = followers_count / account_age_days
    friends_per_day    = friends_count / account_age_days
    listed_per_day     = karma / account_age_days

    # follower_friend_ratio（分母为 0 时返回 0）
    follower_friend_ratio = followers_count / friends_count if friends_count > 0 else 0.0

    # 用户名特征
    screen_name        = normalize_text(author)
    screen_name_length = float(len(screen_name))
    screen_name_digits = float(count_digits(screen_name))

    # bigram likelihood（复用 Weibo 训练集统计）
    bigram_likelihood = compute_bigram_likelihood(screen_name, bigram_stats)

    return [
        sc,                         # 0:  statuses_count
        followers_count,            # 1:  followers_count
        friends_count,              # 2:  friends_count
        karma,                      # 3:  listed_count_proxy_interactions_count
        0.0,                        # 4:  has_custom_profile_image（固定为 0：Moltbook 账号均使用默认头像）
        verified,                   # 5:  verified
        account_age_days,           # 6:  account_age_days
        statuses_per_day,           # 7:  statuses_per_day
        followers_per_day,          # 8:  followers_per_day
        friends_per_day,            # 9:  friends_per_day
        listed_per_day,             # 10: listed_per_day_proxy
        follower_friend_ratio,      # 11: follower_friend_ratio
        screen_name_length,         # 12: screen_name_length
        screen_name_digits,         # 13: screen_name_digits
        screen_name_length,         # 14: name_length_proxy (同 12)
        screen_name_digits,         # 15: name_digits_proxy (同 13)
        float(len(description)),    # 16: description_length
        bigram_likelihood,          # 17: screen_name_bigram_likelihood
    ]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")
    weibo_proc_dir = Path(args.weibo_processed_dir)

    print("=== SGBot Moltbook OOD Preprocessing ===")
    print(f"Data dir          : {DATA_DIR}")
    print(f"Weibo processed   : {weibo_proc_dir}")
    print(f"Reference date    : {args.reference_date}")

    # 1. 加载 bigram_stats（来自 Weibo 训练集，不在 Moltbook 上重新统计）
    print("\n[1/4] Loading bigram_stats from Weibo training set...")
    bigram_stats_path = weibo_proc_dir / "bigram_stats.json"
    if not bigram_stats_path.exists():
        raise FileNotFoundError(
            f"bigram_stats.json not found: {bigram_stats_path}\n"
            "请先运行 baselines-weibo/SGBot/preprocess.py 生成 bigram 统计。"
        )
    bigram_stats = load_json(bigram_stats_path)
    print(
        f"  Loaded bigram_stats: {bigram_stats['observed_bigrams']} bigrams, "
        f"default_prob={bigram_stats['default_probability']:.6e}"
    )

    # 2. 加载用户信息
    print("\n[2/4] Loading user_info.jsonl...")
    user_info = load_user_info()
    ordered_authors = sorted(user_info.keys())  # 确定性排序
    print(f"  {len(ordered_authors)} users loaded")

    # 3. 统计历史帖子数
    print("\n[3/4] Counting history posts (statuses_count approximation)...")
    statuses_counts = load_statuses_count(set(ordered_authors))
    covered = sum(1 for a in ordered_authors if a in statuses_counts)
    print(f"  {covered}/{len(ordered_authors)} users have history posts")

    # 4. 提取 18 维特征
    print("\n[4/4] Extracting 18-dim handcrafted features...")
    rows = []
    for author in ordered_authors:
        sc = statuses_counts.get(author, 0)
        row = extract_feature_row(
            author, user_info[author], sc, reference_time, bigram_stats
        )
        rows.append(row)

    features_arr = np.array(rows, dtype=np.float32)  # [N, 18]
    print(f"  Feature matrix shape: {features_arr.shape}")

    # 保存
    torch.save(
        torch.tensor(features_arr, dtype=torch.float32),
        PROC_DIR / "features.pt",
    )
    save_json(ordered_authors, PROC_DIR / "user_list.json")
    save_json(FEATURE_NAMES, PROC_DIR / "feature_names.json")
    save_json(
        {
            "reference_date": args.reference_date,
            "weibo_processed_dir": str(weibo_proc_dir),
            "n_users": len(ordered_authors),
            "feature_names": FEATURE_NAMES,
            "notes": {
                "has_custom_profile_image": "fixed to 0 (no avatar field in Moltbook)",
                "statuses_count": "approximated by user_history_posts.jsonl row count",
                "bigram_stats": "reused from Weibo training set, not recomputed on Moltbook",
                "join_time_format": "YYYY-MM-DD (date only, no time component)",
                "description_field": "discription (Moltbook original spelling)",
            },
        },
        PROC_DIR / "metadata.json",
    )

    print(f"\nSaved processed data to: {PROC_DIR}")
    print("  features.pt, user_list.json, feature_names.json, metadata.json")


if __name__ == "__main__":
    main()
