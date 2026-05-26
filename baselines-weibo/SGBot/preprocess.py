import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Weibo"
SPLIT_PATH = DATA_DIR / "split.json"
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

LABEL_MAPPING = {0: 0, 2: 1}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Weibo data for SGBot")
    parser.add_argument("--reference_date", type=str, default="2026-04-01")
    parser.add_argument("--device", type=str, default="auto", help="cuda / mps / cpu / auto")
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


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_int(value, default=0):
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def count_digits(value):
    return sum(1 for char in value if char.isdigit())


def compute_account_age_days(created_at, reference_time):
    created_time = parse_datetime(created_at)
    if created_time is None:
        return 1.0
    delta_days = (reference_time - created_time).total_seconds() / 86400.0
    return max(delta_days, 1.0)


def build_split_from_official(labeled_uids):
    raw_split = load_json(SPLIT_PATH)
    official_split = {str(uid): str(split_name).lower() for uid, split_name in raw_split.items()}
    split_indices = {"train": [], "val": [], "test": []}
    missing_uids = []
    unknown_splits = set()

    for index, uid in enumerate(labeled_uids):
        split_name = official_split.get(uid)
        if split_name is None:
            missing_uids.append(uid)
            continue
        if split_name not in split_indices:
            unknown_splits.add(split_name)
            continue
        split_indices[split_name].append(index)

    if unknown_splits:
        raise ValueError(f"Unsupported split names in {SPLIT_PATH}: {sorted(unknown_splits)}")
    if missing_uids:
        preview = ", ".join(missing_uids[:10])
        raise ValueError(
            f"{len(missing_uids)} labeled users are missing from {SPLIT_PATH}. Example uids: {preview}"
        )

    empty_splits = [name for name, indices in split_indices.items() if not indices]
    if empty_splits:
        raise ValueError(f"Official split has empty partitions for labeled users: {empty_splits}")

    return split_indices["train"], split_indices["val"], split_indices["test"]


def get_valid_profiles(label_dict):
    profiles = {}
    valid_labels = {uid: label for uid, label in label_dict.items() if label != 1}
    for item in iter_jsonl(DATA_DIR / "user_profile.jsonl"):
        uid = str(item.get("uid", ""))
        if uid and uid in valid_labels:
            profiles[uid] = item
    return profiles, valid_labels


def build_bigram_stats(screen_names):
    bigram_counter = Counter()
    total_bigrams = 0

    for screen_name in screen_names:
        normalized = normalize_text(screen_name)
        if len(normalized) < 2:
            continue
        for index in range(len(normalized) - 1):
            bigram = normalized[index:index + 2]
            bigram_counter[bigram] += 1
            total_bigrams += 1

    vocab_size = max(len(bigram_counter), 1)
    denominator = total_bigrams + vocab_size + 1
    default_probability = 1.0 / denominator
    probabilities = {
        bigram: (count + 1.0) / denominator for bigram, count in bigram_counter.items()
    }

    return {
        "probabilities": probabilities,
        "default_probability": default_probability,
        "total_bigrams": total_bigrams,
        "observed_bigrams": len(bigram_counter),
    }


def compute_bigram_likelihood(screen_name, bigram_stats):
    normalized = normalize_text(screen_name)
    if len(normalized) < 2:
        return 0.0

    default_probability = bigram_stats["default_probability"]
    probabilities = bigram_stats["probabilities"]
    log_sum = 0.0
    count = 0

    for index in range(len(normalized) - 1):
        bigram = normalized[index:index + 2]
        probability = probabilities.get(bigram, default_probability)
        log_sum += math.log(probability)
        count += 1

    return float(math.exp(log_sum / count)) if count > 0 else 0.0


def extract_feature_row(profile, reference_time, bigram_stats):
    followers_count = float(safe_int(profile.get("followers_count")))
    friends_count = float(safe_int(profile.get("friends_count")))
    statuses_count = float(safe_int(profile.get("statuses_count")))
    listed_count = float(safe_int(profile.get("interactions_count")))
    has_custom_profile_image = float(not safe_bool(profile.get("default_profile_image")))
    verified = float(safe_bool(profile.get("verified")))

    screen_name = normalize_text(profile.get("screen_name"))
    description = normalize_text(profile.get("description"))
    account_age_days = compute_account_age_days(profile.get("created_at"), reference_time)

    if friends_count == 0:
        follower_friend_ratio = 0.0
    else:
        follower_friend_ratio = followers_count / friends_count

    return [
        statuses_count,
        followers_count,
        friends_count,
        listed_count,
        has_custom_profile_image,
        verified,
        account_age_days,
        statuses_count / account_age_days,
        followers_count / account_age_days,
        friends_count / account_age_days,
        listed_count / account_age_days,
        follower_friend_ratio,
        float(len(screen_name)),
        float(count_digits(screen_name)),
        float(len(screen_name)),
        float(count_digits(screen_name)),
        float(len(description)),
        compute_bigram_likelihood(screen_name, bigram_stats),
    ]


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    device = detect_device(args.device)
    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")

    print("=== SGBot Weibo Preprocessing ===")
    print(f"Detected device: {device}")
    print("Current preprocessing is CPU-oriented; device detection is reported for consistency.")

    print("\n[1/5] Loading labels and profiles...")
    label_dict = {str(uid): int(label) for uid, label in load_json(DATA_DIR / "Weibo_Labels.json").items()}
    profiles, valid_labels = get_valid_profiles(label_dict)

    labeled_uids = []
    labels = []
    agent_uids = []
    missing_profiles = []

    for uid, label in valid_labels.items():
        if uid not in profiles:
            missing_profiles.append(uid)
            continue
        if label in LABEL_MAPPING:
            labeled_uids.append(uid)
            labels.append(LABEL_MAPPING[label])
        elif label == -1:
            agent_uids.append(uid)

    print(
        f"  labeled users: {len(labeled_uids)} "
        f"(human={labels.count(0)}, bot={labels.count(1)})"
    )
    print(f"  agent users:   {len(agent_uids)}")
    print(f"  skipped users without profile: {len(missing_profiles)}")

    print("\n[2/5] Building screen-name bigram statistics...")
    bigram_source_uids = labeled_uids + agent_uids
    bigram_stats = build_bigram_stats(
        [profiles[uid].get("screen_name", "") for uid in bigram_source_uids]
    )
    save_json(
        PROC_DIR / "bigram_stats.json",
        {
            "default_probability": bigram_stats["default_probability"],
            "total_bigrams": bigram_stats["total_bigrams"],
            "observed_bigrams": bigram_stats["observed_bigrams"],
            "probabilities": bigram_stats["probabilities"],
        },
    )
    print(
        f"  observed bigrams: {bigram_stats['observed_bigrams']}, "
        f"total bigrams: {bigram_stats['total_bigrams']}"
    )

    print("\n[3/5] Extracting handcrafted features...")
    labeled_features = np.asarray(
        [extract_feature_row(profiles[uid], reference_time, bigram_stats) for uid in labeled_uids],
        dtype=np.float32,
    )
    agent_features = np.asarray(
        [extract_feature_row(profiles[uid], reference_time, bigram_stats) for uid in agent_uids],
        dtype=np.float32,
    )
    labels_array = np.asarray(labels, dtype=np.int64)
    print(f"  labeled feature shape: {labeled_features.shape}")
    print(f"  agent feature shape:   {agent_features.shape}")

    print("\n[4/5] Building train/val/test split from official split.json...")
    train_idx, val_idx, test_idx = build_split_from_official(labeled_uids)
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print("\n[5/5] Saving processed artifacts...")
    torch.save(torch.tensor(labeled_features, dtype=torch.float32).cpu(), PROC_DIR / "features.pt")
    torch.save(torch.tensor(labels_array, dtype=torch.long).cpu(), PROC_DIR / "labels.pt")
    torch.save(torch.tensor(train_idx, dtype=torch.long).cpu(), PROC_DIR / "train_idx.pt")
    torch.save(torch.tensor(val_idx, dtype=torch.long).cpu(), PROC_DIR / "val_idx.pt")
    torch.save(torch.tensor(test_idx, dtype=torch.long).cpu(), PROC_DIR / "test_idx.pt")
    torch.save(torch.tensor(agent_features, dtype=torch.float32).cpu(), PROC_DIR / "agent_features.pt")

    save_json(PROC_DIR / "feature_names.json", FEATURE_NAMES)
    save_json(PROC_DIR / "labeled_uids.json", labeled_uids)
    save_json(PROC_DIR / "agent_uids.json", agent_uids)
    save_json(
        PROC_DIR / "metadata.json",
        {
            "reference_date": args.reference_date,
            "split_source": str(SPLIT_PATH),
            "split_strategy": "official",
            "feature_names": FEATURE_NAMES,
            "label_mapping": {"human": 0, "bot": 1},
            "raw_label_policy": {
                "0": "keep_as_human",
                "2": "keep_as_bot",
                "1": "drop_completely",
                "-1": "ood_only",
            },
            "missing_profile_count": len(missing_profiles),
            "labeled_count": len(labeled_uids),
            "agent_count": len(agent_uids),
        },
    )

    print(f"Saved processed data to: {PROC_DIR}")


if __name__ == "__main__":
    main()