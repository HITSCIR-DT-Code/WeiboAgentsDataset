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


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Weibo"
MODEL_DIR = ROOT_DIR / "models" / "xlm-roberta-base"
SPLIT_PATH = DATA_DIR / "split.json"
SAVE_DIR = BASE_DIR / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "friends_count",
    "bidirectional_links_ratio",
    "statuses_count",
    "account_age_seconds",
    "url_ratio",
    "unique_url_ratio",
    "tweet_similarity",
    "avg_followee_followers",
    "avg_followee_statuses",
    "friends_to_median_followee_followers",
    "friends_rate",
]
LABEL_MAPPING = {0: 0, 2: 1}
MAX_TWEETS_PER_USER = 20
URL_PREFIXES = ("http://", "https://", "www.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Weibo data for EvolveBot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference_date", type=str, default="2026-04-01")
    parser.add_argument("--device", type=str, default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def detect_device(requested):
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def iterate_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_int(value, default=0):
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def extract_urls(text):
    urls = []
    for token in text.split():
        token_lower = token.lower()
        if token_lower.startswith(URL_PREFIXES):
            urls.append(token)
    return urls


def load_profiles(valid_uids):
    profiles = {}
    for item in iterate_jsonl(DATA_DIR / "user_profile.jsonl"):
        uid = str(item.get("uid", ""))
        if uid in valid_uids:
            profiles[uid] = item
    return profiles


def load_follow_graph(valid_uids):
    followings = defaultdict(set)
    for item in iterate_jsonl(DATA_DIR / "followings.jsonl"):
        source_uid = str(item.get("source_uid", ""))
        target_uid = str(item.get("target_uid", ""))
        if source_uid in valid_uids and target_uid in valid_uids and source_uid != target_uid:
            followings[source_uid].add(target_uid)
    return followings


def load_recent_tweets(valid_uids):
    tweets_by_uid = defaultdict(list)
    for item in iterate_jsonl(DATA_DIR / "user_tweets.jsonl"):
        uid = str(item.get("uid", ""))
        if uid not in valid_uids:
            continue
        text = (item.get("text") or "").strip()
        created_at = parse_datetime(item.get("created_at"))
        tweets_by_uid[uid].append((created_at, text))

    recent_tweets = {}
    for uid, items in tweets_by_uid.items():
        items.sort(key=lambda value: value[0] or datetime.min, reverse=True)
        texts = []
        for _, text in items:
            if text:
                texts.append(text)
            if len(texts) == MAX_TWEETS_PER_USER:
                break
        recent_tweets[uid] = texts
    return recent_tweets


def load_official_split(labeled_uids):
    split_map = {str(uid): split for uid, split in load_json(SPLIT_PATH).items()}
    split_uids = {"train": [], "val": [], "test": []}
    missing_uids = []

    for uid in labeled_uids:
        split_name = split_map.get(uid)
        if split_name not in split_uids:
            missing_uids.append(uid)
            continue
        split_uids[split_name].append(uid)

    if missing_uids:
        preview = ", ".join(missing_uids[:5])
        raise ValueError(
            f"Official split is missing {len(missing_uids)} labeled users with valid profiles. Examples: {preview}"
        )

    for split_name, uids in split_uids.items():
        if not uids:
            raise ValueError(f"Official split '{split_name}' is empty after filtering valid labeled users")

    return split_uids


def compute_graph_statistics(valid_uids, profiles, followings):
    graph_stats = {}
    for uid in valid_uids:
        neighbors = followings.get(uid, set())
        out_degree = len(neighbors)
        reciprocal = 0
        neighbor_followers = []
        neighbor_statuses = []

        for neighbor_uid in neighbors:
            if uid in followings.get(neighbor_uid, set()):
                reciprocal += 1
            neighbor_profile = profiles.get(neighbor_uid)
            if neighbor_profile is None:
                continue
            neighbor_followers.append(safe_int(neighbor_profile.get("followers_count")))
            neighbor_statuses.append(safe_int(neighbor_profile.get("statuses_count")))

        graph_stats[uid] = {
            "out_degree": out_degree,
            "reciprocal": reciprocal,
            "neighbor_followers": neighbor_followers,
            "neighbor_statuses": neighbor_statuses,
        }
    return graph_stats


def build_text_embeddings(texts_by_uid, tokenizer, model, device, batch_size, max_length):
    unique_texts = []
    for texts in texts_by_uid.values():
        unique_texts.extend(texts)
    unique_texts = list(dict.fromkeys(unique_texts))

    embedding_map = {}
    if not unique_texts:
        return embedding_map

    model.eval()
    for start in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding tweets"):
        batch_texts = unique_texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        for text, embedding in zip(batch_texts, embeddings):
            embedding_map[text] = embedding
    return embedding_map


def compute_tweet_similarity(texts, embedding_map):
    if len(texts) <= 1:
        return 0.0

    embeddings = []
    for text in texts:
        embedding = embedding_map.get(text)
        if embedding is None:
            continue
        embeddings.append(embedding)
    if len(embeddings) <= 1:
        return 0.0

    matrix = torch.stack(embeddings, dim=0).float()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    cosine = matrix @ matrix.t()
    upper_indices = torch.triu_indices(cosine.size(0), cosine.size(1), offset=1)
    values = cosine[upper_indices[0], upper_indices[1]]
    return float(values.mean().item()) if values.numel() > 0 else 0.0


def extract_feature_row(uid, profiles, graph_stats, tweets_by_uid, embedding_map, reference_time):
    profile = profiles[uid]
    follow_count = float(safe_int(profile.get("friends_count")))
    statuses_count = float(safe_int(profile.get("statuses_count")))
    created_at = parse_datetime(profile.get("created_at"))
    account_age_seconds = 0.0
    if created_at is not None:
        account_age_seconds = max((reference_time - created_at).total_seconds(), 1.0)

    texts = tweets_by_uid.get(uid, [])
    tweet_count = len(texts)
    url_count = 0
    unique_urls = set()
    for text in texts:
        urls = extract_urls(text)
        url_count += len(urls)
        unique_urls.update(urls)

    stats = graph_stats.get(uid, {})
    out_degree = stats.get("out_degree", 0)
    reciprocal = stats.get("reciprocal", 0)
    neighbor_followers = stats.get("neighbor_followers", [])
    neighbor_statuses = stats.get("neighbor_statuses", [])

    bidirectional_ratio = reciprocal / out_degree if out_degree else 0.0
    avg_followee_followers = float(np.mean(neighbor_followers)) if neighbor_followers else 0.0
    avg_followee_statuses = float(np.mean(neighbor_statuses)) if neighbor_statuses else 0.0
    median_followee_followers = statistics.median(neighbor_followers) if neighbor_followers else 0.0
    friends_to_median_followee_followers = (
        follow_count / median_followee_followers if median_followee_followers else 0.0
    )
    friends_rate = follow_count / account_age_seconds if account_age_seconds else 0.0
    url_ratio = url_count / tweet_count if tweet_count else 0.0
    unique_url_ratio = len(unique_urls) / tweet_count if tweet_count else 0.0
    tweet_similarity = compute_tweet_similarity(texts, embedding_map)

    return [
        follow_count,
        bidirectional_ratio,
        statuses_count,
        float(account_age_seconds),
        url_ratio,
        unique_url_ratio,
        tweet_similarity,
        avg_followee_followers,
        avg_followee_statuses,
        friends_to_median_followee_followers,
        friends_rate,
    ]


def zscore_by_train(train_array, other_arrays):
    mean = train_array.mean(axis=0)
    std = train_array.std(axis=0)
    std[std == 0] = 1.0
    normalized = [(array - mean) / std for array in [train_array] + list(other_arrays)]
    return normalized, mean, std


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")
    device = detect_device(args.device)

    print("=== EvolveBot Weibo Preprocessing ===")
    print(f"Data dir: {DATA_DIR}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Split file: {SPLIT_PATH}")
    print(f"Device: {device}")
    print(f"Reference date: {reference_time.strftime('%Y-%m-%d')}")

    label_dict = {str(uid): int(label) for uid, label in load_json(DATA_DIR / "Weibo_Labels.json").items()}
    valid_uids = [uid for uid, label in label_dict.items() if label in (-1, 0, 2)]
    profiles = load_profiles(set(valid_uids))

    labeled_uids = []
    labeled_labels = []
    agent_uids = []
    for uid in valid_uids:
        if uid not in profiles:
            continue
        label = label_dict[uid]
        if label in LABEL_MAPPING:
            labeled_uids.append(uid)
            labeled_labels.append(LABEL_MAPPING[label])
        elif label == -1:
            agent_uids.append(uid)

    print(f"Labeled users: {len(labeled_uids)} (human={labeled_labels.count(0)}, bot={labeled_labels.count(1)})")
    print(f"Agent users: {len(agent_uids)}")

    label_by_uid = dict(zip(labeled_uids, labeled_labels))
    split_uids = load_official_split(labeled_uids)
    train_uids = split_uids["train"]
    val_uids = split_uids["val"]
    test_uids = split_uids["test"]
    ordered_labeled_uids = train_uids + val_uids + test_uids
    ordered_labels = [label_by_uid[uid] for uid in ordered_labeled_uids]

    graph_uids = set(ordered_labeled_uids + agent_uids)
    print("Loading follow graph and tweets...")
    followings = load_follow_graph(graph_uids)
    tweets_by_uid = load_recent_tweets(graph_uids)
    graph_stats = compute_graph_statistics(graph_uids, profiles, followings)

    print("Loading text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device)
    embedding_map = build_text_embeddings(
        tweets_by_uid,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )
    del model

    print("Extracting handcrafted features...")
    labeled_raw = np.array(
        [
            extract_feature_row(uid, profiles, graph_stats, tweets_by_uid, embedding_map, reference_time)
            for uid in ordered_labeled_uids
        ],
        dtype=np.float32,
    )
    if agent_uids:
        agent_raw = np.array(
            [
                extract_feature_row(uid, profiles, graph_stats, tweets_by_uid, embedding_map, reference_time)
                for uid in agent_uids
            ],
            dtype=np.float32,
        )
    else:
        agent_raw = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    train_count = len(train_uids)
    val_count = len(val_uids)
    train_raw = labeled_raw[:train_count]
    val_raw = labeled_raw[train_count:train_count + val_count]
    test_raw = labeled_raw[train_count + val_count:]
    [train_norm, val_norm, test_norm, agent_norm], mean, std = zscore_by_train(
        train_raw,
        [val_raw, test_raw, agent_raw],
    )
    labeled_features = np.concatenate([train_norm, val_norm, test_norm], axis=0)

    labels_tensor = torch.tensor(ordered_labels, dtype=torch.long).cpu()
    feature_tensor = torch.tensor(labeled_features, dtype=torch.float32).cpu()
    agent_feature_tensor = torch.tensor(agent_norm, dtype=torch.float32).cpu()
    train_idx_tensor = torch.arange(0, train_count, dtype=torch.long).cpu()
    val_idx_tensor = torch.arange(train_count, train_count + val_count, dtype=torch.long).cpu()
    test_idx_tensor = torch.arange(train_count + val_count, len(ordered_labeled_uids), dtype=torch.long).cpu()

    torch.save(feature_tensor, SAVE_DIR / "features.pt")
    torch.save(labels_tensor, SAVE_DIR / "labels.pt")
    torch.save(train_idx_tensor, SAVE_DIR / "train_idx.pt")
    torch.save(val_idx_tensor, SAVE_DIR / "val_idx.pt")
    torch.save(test_idx_tensor, SAVE_DIR / "test_idx.pt")
    torch.save(agent_feature_tensor, SAVE_DIR / "agent_features.pt")

    save_json(ordered_labeled_uids, SAVE_DIR / "labeled_uids.json")
    save_json(agent_uids, SAVE_DIR / "agent_uids.json")
    save_json(FEATURE_NAMES, SAVE_DIR / "feature_names.json")
    save_json(
        {
            "seed": args.seed,
            "split_source": str(SPLIT_PATH),
            "reference_date": args.reference_date,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "num_labeled": len(ordered_labeled_uids),
            "num_agent": len(agent_uids),
        },
        SAVE_DIR / "metadata.json",
    )

    print("Saved files:")
    print(f"  {SAVE_DIR / 'features.pt'}")
    print(f"  {SAVE_DIR / 'labels.pt'}")
    print(f"  {SAVE_DIR / 'agent_features.pt'}")
    print(f"  train/val/test = {len(train_idx_tensor)}/{len(val_idx_tensor)}/{len(test_idx_tensor)}")


if __name__ == "__main__":
    main()