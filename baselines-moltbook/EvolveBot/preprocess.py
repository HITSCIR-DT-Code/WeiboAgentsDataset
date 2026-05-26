"""
  cd baselines-moltbook/EvolveBot
  python preprocess.py
  python preprocess.py --device cpu --batch_size 32
  python preprocess.py --weibo_processed_dir ../../baselines-weibo/EvolveBot/processed
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Moltbook"
MODEL_DIR = ROOT_DIR / "models" / "xlm-roberta-base"
SAVE_DIR = BASE_DIR / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NOGRAPH_INDICES = [0, 2, 3, 4, 5, 6, 10]
NOGRAPH_NAMES = [
    "friends_count",
    "statuses_count",
    "account_age_seconds",
    "url_ratio",
    "unique_url_ratio",
    "tweet_similarity",
    "friends_rate",
]
MAX_POSTS_PER_USER = 20
URL_PREFIXES = ("http://", "https://", "www.")
REFERENCE_DATE = "2026-04-01"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Moltbook data for EvolveBot OOD")
    parser.add_argument(
        "--weibo_processed_dir",
        type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "EvolveBot" / "processed"),
        help="Weibo EvolveBot processed 目录，用于读取归一化参数",
    )
    parser.add_argument(
        "--reference_date", type=str, default=REFERENCE_DATE,
        help="账号年龄截止日期 (YYYY-MM-DD)"
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def detect_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
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


def parse_datetime_full(value: str):
    """Parse '2026-03-09 00:00:00' format."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_date_only(value: str):
    """Parse '2026-02-11' format (join_time in user_info)."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def extract_urls(text: str):
    urls = []
    for token in text.split():
        if token.lower().startswith(URL_PREFIXES):
            urls.append(token)
    return urls


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_user_info() -> dict:
    """Returns {author: info_dict}."""
    users = {}
    for item in iterate_jsonl(DATA_DIR / "user_info.jsonl"):
        author = item.get("author", "")
        if author:
            users[author] = item
    return users


def load_history_posts(known_authors: set) -> dict:
    """
    Returns {author: [{"time": datetime, "short_content": str}, ...]},
    sorted by time descending, truncated to MAX_POSTS_PER_USER.
    Also returns statuses_count (total rows seen) per author.
    """
    raw: dict[str, list] = defaultdict(list)
    statuses_count: dict[str, int] = defaultdict(int)

    for item in iterate_jsonl(DATA_DIR / "user_history_posts.jsonl"):
        author = item.get("author", "")
        if author not in known_authors:
            continue
        statuses_count[author] += 1
        t = parse_datetime_full(item.get("time", ""))
        content = (item.get("short_content") or "").strip()
        raw[author].append((t, content))

    recent: dict[str, list] = {}
    for author, items in raw.items():
        items.sort(key=lambda x: x[0] or datetime.min, reverse=True)
        texts = []
        for _, content in items:
            if content:
                texts.append(content)
            if len(texts) == MAX_POSTS_PER_USER:
                break
        recent[author] = texts

    return recent, dict(statuses_count)


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------
def build_text_embeddings(texts_by_author: dict, tokenizer, model, device, batch_size, max_length) -> dict:
    unique_texts = []
    for texts in texts_by_author.values():
        unique_texts.extend(texts)
    unique_texts = list(dict.fromkeys(unique_texts))

    embedding_map = {}
    if not unique_texts:
        return embedding_map

    model.eval()
    for start in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding posts"):
        batch = unique_texts[start: start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        for text, emb in zip(batch, embeddings):
            embedding_map[text] = emb
    return embedding_map


def compute_tweet_similarity(texts: list, embedding_map: dict) -> float:
    if len(texts) <= 1:
        return 0.0
    embeddings = [embedding_map[t] for t in texts if t in embedding_map]
    if len(embeddings) <= 1:
        return 0.0
    matrix = torch.stack(embeddings, dim=0).float()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    cosine = matrix @ matrix.t()
    idx = torch.triu_indices(cosine.size(0), cosine.size(1), offset=1)
    vals = cosine[idx[0], idx[1]]
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_feature_row(author: str, user_info: dict, recent_posts: list,
                        statuses_count: int, embedding_map: dict,
                        reference_time: datetime) -> list:
    info = user_info[author]

    # friends_count (following in Moltbook)
    friends_count = float(safe_int(info.get("following")))

    # statuses_count — total posts crawled (best approximation available)
    sc = float(statuses_count)

    # account_age_seconds
    join_dt = parse_date_only(info.get("join_time", ""))
    if join_dt is not None:
        account_age_seconds = max((reference_time - join_dt).total_seconds(), 1.0)
    else:
        account_age_seconds = 1.0

    # url_ratio / unique_url_ratio from short_content of recent posts
    tweet_count = len(recent_posts)
    url_count = 0
    unique_urls: set = set()
    for text in recent_posts:
        urls = extract_urls(text)
        url_count += len(urls)
        unique_urls.update(urls)
    url_ratio = url_count / tweet_count if tweet_count else 0.0
    unique_url_ratio = len(unique_urls) / tweet_count if tweet_count else 0.0

    # tweet_similarity
    tweet_similarity = compute_tweet_similarity(recent_posts, embedding_map)

    # friends_rate
    friends_rate = friends_count / account_age_seconds

    # Order must match NOGRAPH_INDICES selection from the full 11-dim vector:
    # [friends_count, statuses_count, account_age_seconds,
    #  url_ratio, unique_url_ratio, tweet_similarity, friends_rate]
    return [
        friends_count,
        sc,
        float(account_age_seconds),
        url_ratio,
        unique_url_ratio,
        tweet_similarity,
        friends_rate,
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")
    device = detect_device(args.device)
    weibo_proc_dir = Path(args.weibo_processed_dir)

    print("=== EvolveBot Moltbook Preprocessing ===")
    print(f"Data dir    : {DATA_DIR}")
    print(f"Weibo proc  : {weibo_proc_dir}")
    print(f"Model dir   : {MODEL_DIR}")
    print(f"Device      : {device}")
    print(f"Ref date    : {args.reference_date}")

    # ----- load normalization params from Weibo training -----
    metadata_path = weibo_proc_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Weibo metadata not found: {metadata_path}\n"
            "Run baselines-weibo/EvolveBot/preprocess.py first."
        )
    metadata = load_json(metadata_path)
    full_mean = np.array(metadata["feature_mean"], dtype=np.float32)   # [11]
    full_std  = np.array(metadata["feature_std"],  dtype=np.float32)   # [11]
    nograph_mean = full_mean[NOGRAPH_INDICES]   # [7]
    nograph_std  = full_std[NOGRAPH_INDICES]    # [7]
    nograph_std[nograph_std == 0] = 1.0

    print(f"Loaded normalization params from {metadata_path}")

    # ----- load user info -----
    print("Loading user_info.jsonl ...")
    user_info = load_user_info()
    all_authors = list(user_info.keys())
    print(f"  {len(all_authors)} users loaded")

    # ----- load history posts -----
    print("Loading user_history_posts.jsonl (streaming) ...")
    recent_posts_by_author, statuses_counts = load_history_posts(set(all_authors))
    print(f"  {len(recent_posts_by_author)} users have history posts")

    # ----- text embeddings -----
    print("Loading text encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device)
    embedding_map = build_text_embeddings(
        recent_posts_by_author, tokenizer, model, device,
        args.batch_size, args.max_length,
    )
    del model

    # ----- extract features -----
    print("Extracting features ...")
    ordered_authors = sorted(all_authors)   # deterministic order
    rows = []
    for author in tqdm(ordered_authors, desc="Users"):
        recent = recent_posts_by_author.get(author, [])
        sc = statuses_counts.get(author, 0)
        row = extract_feature_row(
            author, user_info, recent, sc, embedding_map, reference_time
        )
        rows.append(row)

    raw_array = np.array(rows, dtype=np.float32)   # [N, 7]
    norm_array = (raw_array - nograph_mean) / nograph_std

    raw_tensor  = torch.tensor(raw_array,  dtype=torch.float32)
    feat_tensor = torch.tensor(norm_array, dtype=torch.float32)

    torch.save(feat_tensor, SAVE_DIR / "features.pt")
    torch.save(raw_tensor,  SAVE_DIR / "raw_features.pt")
    save_json(ordered_authors, SAVE_DIR / "user_list.json")
    save_json(NOGRAPH_NAMES,   SAVE_DIR / "feature_names.json")

    print(f"\nSaved {len(ordered_authors)} users, feature shape: {feat_tensor.shape}")
    print(f"  {SAVE_DIR / 'features.pt'}")
    print(f"  {SAVE_DIR / 'raw_features.pt'}")
    print(f"  {SAVE_DIR / 'user_list.json'}")
    print(f"  {SAVE_DIR / 'feature_names.json'}")


if __name__ == "__main__":
    main()
