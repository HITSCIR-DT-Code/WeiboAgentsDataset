"""
data_loader.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── 路径 ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
WEIBO_DIR = BASE_DIR / "dataset" / "Weibo"
MOLTBOOK_DIR = BASE_DIR / "dataset" / "Moltbook"

LABELS_FILE   = WEIBO_DIR / "Weibo_Labels.json"
PROFILE_FILE  = WEIBO_DIR / "user_profile.jsonl"
TWEETS_FILE   = WEIBO_DIR / "user_tweets.jsonl"
MOLT_INFO_FILE  = MOLTBOOK_DIR / "user_info.jsonl"
MOLT_POSTS_FILE = MOLTBOOK_DIR / "user_history_posts.jsonl"

REFERENCE_DATE = datetime(2026, 4, 1)

# ── 工具 ─────────────────────────────────────────────────────────────────────
def _age_days(date_str: str, fmt="%Y-%m-%d %H:%M:%S") -> float:
    """返回账号年龄（天数），解析失败返回 None。"""
    for f in (fmt, "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), f)
            delta = (REFERENCE_DATE - dt).total_seconds() / 86400
            return max(delta, 1.0)   # 至少 1 天，避免除零
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


# ── 核心加载 ──────────────────────────────────────────────────────────────────
def load_weibo_groups(verbose: bool = True):
    """
    加载微博数据，返回：
            {"Human": [user_dict, ...], "Bot": [...], "Agent(Weibo)": [...]}
    """
    # 1. 标签
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels: dict[str, int] = json.load(f)

    label_map = {}   # uid → group_name
    for uid, lbl in labels.items():
        if lbl == 0:
            label_map[uid] = "Human"
        elif lbl == 2:
            label_map[uid] = "Bot"
        elif lbl == -1:
            label_map[uid] = "Agent(Weibo)"
        # label=1 疑似，跳过

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
                "followers_count": rec.get("followers_count", 0) or 0,
                "friends_count":   rec.get("friends_count",   0) or 0,
                "statuses_count":  rec.get("statuses_count",  0) or 0,
                "interactions_count": rec.get("interactions_count", 0) or 0,
                "verified":              bool(rec.get("verified", False)),
                "default_profile_image": bool(rec.get("default_profile_image", True)),
                "description": rec.get("description", "") or "",
                "account_age_days": age,
                "tweets": [],   # 待填充
            }

    # 3. 流式读取推文，按 uid 聚合
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
                "text":       rec.get("text", "") or "",
                "ts":         ts,
                "is_retweet": bool(rec.get("is_retweet", False)),
                "has_image":  bool(rec.get("images")),
                "comments_count": rec.get("comments_count", 0) or 0,
            })

    for uid, twts in tweet_buckets.items():
        if uid in profiles:
            # 按时间排序（便于计算间隔）
            profiles[uid]["tweets"] = sorted(
                twts, key=lambda x: x["ts"] or datetime.min
            )

    # 4. 按 group 分组
    groups: dict[str, list] = {"Human": [], "Bot": [], "Agent(Weibo)": []}
    for uid, pdata in profiles.items():
        g = pdata["label_group"]
        if g in groups:
            groups[g].append(pdata)

    if verbose:
        for g, lst in groups.items():
            print(f"  Weibo {g:14s}: {len(lst)} users")

    return groups


def load_moltbook_agents(verbose: bool = True):
    """
    加载 Moltbook 数据，返回：
      [user_dict, ...]
    每个 user_dict 包含 author, followers, following, karma, verified,
    description, account_age_days, posts=[{"text","ts"}]
    """
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
                "author":   author,
                "followers": rec.get("followers", 0) or 0,
                "following": rec.get("following", 0) or 0,
                "karma":    rec.get("karma", 0) or 0,
                "verified": bool(rec.get("verified", False)),
                "description": rec.get("discription", "") or "",
                "account_age_days": age,
                "posts": [],
            }

    # 2. 流式读取历史帖子
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
                "text": text,
                "ts": ts,
                "comment_counts": rec.get("comment_counts", 0) or 0,
                "vote_count": rec.get("vote_count", 0) or 0,
            })

    # 按时间排序
    for u in users.values():
        u["posts"] = sorted(u["posts"], key=lambda x: x["ts"] or datetime.min)

    result = list(users.values())
    if verbose:
        print(f"  Moltbook agent: {len(result)} users")

    return result


# ── 独立运行（验证样本量）────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Weibo groups...")
    groups = load_weibo_groups()
    print("Loading Moltbook agents...")
    agents = load_moltbook_agents()
    print("Done.")
