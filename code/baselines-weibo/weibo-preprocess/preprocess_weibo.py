import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

"""
python preprocess_weibo.py
"""

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Weibo"
MODEL_DIR = ROOT_DIR / "models" / "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = BASE_DIR / "weibo_processed_data"
SPLIT_PATH = DATA_DIR / "split.json"
REFERENCE_DATE = "2026-04-01"
AGENT_LABEL = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Weibo data into the TwiBot-20/GCN_GAT compatible format."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--reference-date", type=str, default=REFERENCE_DATE)
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-tweets", type=int, default=20)
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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def iterate_jsonl(path):
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


def load_profiles():
    profiles = {}
    for item in iterate_jsonl(DATA_DIR / "user_profile.jsonl"):
        uid = str(item.get("uid", ""))
        if uid:
            profiles[uid] = item
    return profiles


def load_recent_tweets(max_tweets):
    tweets_by_uid = {}
    for item in iterate_jsonl(DATA_DIR / "user_tweets.jsonl"):
        uid = str(item.get("uid", ""))
        if not uid:
            continue
        text = (item.get("text") or "").strip()
        created_at = parse_datetime(item.get("created_at"))
        tweets_by_uid.setdefault(uid, []).append((created_at, text))

    recent_tweets = {}
    for uid, items in tweets_by_uid.items():
        items.sort(key=lambda pair: pair[0] or datetime.min, reverse=True)
        texts = []
        for _, text in items:
            if text:
                texts.append(text)
            if len(texts) >= max_tweets:
                break
        recent_tweets[uid] = texts
    return recent_tweets


def load_followings():
    edges = []
    for item in iterate_jsonl(DATA_DIR / "followings.jsonl"):
        source_uid = str(item.get("source_uid", ""))
        target_uid = str(item.get("target_uid", ""))
        if source_uid and target_uid:
            edges.append((source_uid, target_uid))
    return edges


def build_node_lists(label_dict, split_dict, profiles, tweets_by_uid, follow_edges):
    labeled_uids = []
    labeled_labels = []
    agent_uids = []
    dropped_missing_profile = []
    dropped_missing_split = []
    split_buckets = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}

    for uid, raw_label in label_dict.items():
        if uid not in profiles:
            dropped_missing_profile.append(uid)
            continue
        if raw_label == 1:
            continue
        if raw_label == 0:
            mapped_label = 0
        elif raw_label == 2:
            mapped_label = 1
        elif raw_label == -1:
            agent_uids.append(uid)
            continue
        else:
            continue

        split_name = split_dict.get(uid)
        if split_name not in split_buckets:
            dropped_missing_split.append(uid)
            continue
        split_buckets[split_name].append(uid)
        split_labels[split_name].append(mapped_label)

    ordered_labeled_uids = split_buckets["train"] + split_buckets["val"] + split_buckets["test"]
    ordered_labeled_labels = split_labels["train"] + split_labels["val"] + split_labels["test"]
    labeled_uids = ordered_labeled_uids
    labeled_labels = ordered_labeled_labels

    all_uids = ordered_labeled_uids + agent_uids
    uid_to_index = {uid: idx for idx, uid in enumerate(all_uids)}

    metadata = {
        "dropped_missing_profile_count": len(dropped_missing_profile),
        "dropped_missing_profile_uids": dropped_missing_profile,
        "dropped_missing_split_count": len(dropped_missing_split),
        "dropped_missing_split_uids": dropped_missing_split,
        "tweet_users_available": len(tweets_by_uid),
        "raw_follow_edge_count": len(follow_edges),
        "num_labeled_users": len(ordered_labeled_uids),
        "num_agent_users": len(agent_uids),
        "num_total_nodes": len(all_uids),
        "num_train": len(split_buckets["train"]),
        "num_val": len(split_buckets["val"]),
        "num_test": len(split_buckets["test"]),
    }
    return all_uids, ordered_labeled_uids, ordered_labeled_labels, agent_uids, uid_to_index, metadata


def build_num_features(uids, profiles, reference_time):
    rows = []
    for uid in uids:
        profile = profiles[uid]
        created_at = parse_datetime(profile.get("created_at"))
        active_days = 1.0
        if created_at is not None:
            active_days = max((reference_time - created_at).days, 1)

        rows.append(
            [
                float(safe_int(profile.get("followers_count"))),
                float(active_days),
                float(len((profile.get("screen_name") or "").strip())),
                float(safe_int(profile.get("friends_count"))),
                float(safe_int(profile.get("interactions_count"))),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def normalize_with_train_stats(train_array, arrays):
    mean = train_array.mean(axis=0)
    std = train_array.std(axis=0)
    std[std == 0] = 1.0
    normalized = []
    for array in arrays:
        if array.size == 0:
            normalized.append(array.astype(np.float32))
        else:
            normalized.append(((array - mean) / std).astype(np.float32))
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def build_cat_features(uids, profiles):
    rows = []
    for uid in uids:
        default_profile_image = float(safe_bool(profiles[uid].get("default_profile_image")))
        rows.append([default_profile_image])
    return np.asarray(rows, dtype=np.float32)


def build_graph_edges(uid_to_index, raw_edges):
    seen = set()
    edge_pairs = []
    for source_uid, target_uid in raw_edges:
        if source_uid not in uid_to_index or target_uid not in uid_to_index:
            continue
        source_idx = uid_to_index[source_uid]
        target_idx = uid_to_index[target_uid]
        if source_idx == target_idx:
            continue
        pair = (source_idx, target_idx)
        if pair in seen:
            continue
        seen.add(pair)
        edge_pairs.append(pair)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_type = torch.zeros(len(edge_pairs), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    return edge_index, edge_type


def get_sentence_embedding(model_outputs):
    if getattr(model_outputs, "pooler_output", None) is not None:
        return model_outputs.pooler_output
    return model_outputs.last_hidden_state[:, 0, :]


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size, max_length, progress_label):
    hidden_size = model.config.hidden_size
    if not texts:
        return torch.empty((0, hidden_size), dtype=torch.float32)

    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=progress_label):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        embeddings.append(get_sentence_embedding(outputs).detach().cpu())
    return torch.cat(embeddings, dim=0)


def encode_descriptions(all_uids, profiles, tokenizer, model, device, batch_size, max_length):
    descriptions = [(profiles[uid].get("description") or "").strip() for uid in all_uids]
    hidden_size = model.config.hidden_size
    desc_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)
    non_empty_indices = [idx for idx, text in enumerate(descriptions) if text]
    non_empty_texts = [descriptions[idx] for idx in non_empty_indices]
    if non_empty_texts:
        encoded = encode_texts(
            non_empty_texts,
            tokenizer,
            model,
            device,
            batch_size,
            max_length,
            "Encoding descriptions",
        )
        desc_tensor[non_empty_indices] = encoded
    return desc_tensor


def encode_tweets(all_uids, tweets_by_uid, tokenizer, model, device, batch_size, max_length):
    hidden_size = model.config.hidden_size
    tweet_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)

    flat_texts = []
    owners = []
    for node_index, uid in enumerate(all_uids):
        for text in tweets_by_uid.get(uid, []):
            if not text:
                continue
            flat_texts.append(text)
            owners.append(node_index)

    if not flat_texts:
        return tweet_tensor

    encoded = encode_texts(
        flat_texts,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        "Encoding tweets",
    )

    counts = torch.zeros(len(all_uids), dtype=torch.float32)
    for emb_index, node_index in enumerate(owners):
        tweet_tensor[node_index] += encoded[emb_index]
        counts[node_index] += 1

    nonzero = counts > 0
    tweet_tensor[nonzero] = tweet_tensor[nonzero] / counts[nonzero].unsqueeze(1)
    return tweet_tensor


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")
    device = detect_device(args.device)

    print("=== Weibo TwiBot-Compatible Preprocessing ===")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    print("\n[1/6] Loading labels, profiles, tweets, and followings...")
    label_dict = {str(uid): int(label) for uid, label in load_json(DATA_DIR / "Weibo_Labels.json").items()}
    split_dict = {str(uid): split for uid, split in load_json(SPLIT_PATH).items()}
    profiles = load_profiles()
    tweets_by_uid = load_recent_tweets(args.max_tweets)
    follow_edges = load_followings()

    print("[2/6] Building node order and dataset splits...")
    all_uids, labeled_uids, labeled_labels, agent_uids, uid_to_index, metadata = build_node_lists(
        label_dict,
        split_dict,
        profiles,
        tweets_by_uid,
        follow_edges,
    )
    num_train = metadata["num_train"]
    num_val = metadata["num_val"]
    num_test = metadata["num_test"]

    print(
        f"  labeled={len(labeled_uids)} (train={num_train}, val={num_val}, test={num_test}), "
        f"agent={len(agent_uids)}, total_nodes={len(all_uids)}"
    )

    print("[3/6] Building graph and structured features...")
    train_uids = labeled_uids[:num_train]
    val_uids = labeled_uids[num_train:num_train + num_val]
    test_uids = labeled_uids[num_train + num_val:num_train + num_val + num_test]

    train_num = build_num_features(train_uids, profiles, reference_time)
    val_num = build_num_features(val_uids, profiles, reference_time)
    test_num = build_num_features(test_uids, profiles, reference_time)
    agent_num = build_num_features(agent_uids, profiles, reference_time)
    normalized_arrays, mean, std = normalize_with_train_stats(
        train_num,
        [train_num, val_num, test_num, agent_num],
    )
    train_num, val_num, test_num, agent_num = normalized_arrays

    num_properties_tensor = torch.tensor(
        np.concatenate([train_num, val_num, test_num, agent_num], axis=0),
        dtype=torch.float32,
    )
    cat_properties_tensor = torch.tensor(build_cat_features(all_uids, profiles), dtype=torch.float32)
    edge_index, edge_type = build_graph_edges(uid_to_index, follow_edges)

    labels_tensor = torch.tensor(
        labeled_labels + [AGENT_LABEL] * len(agent_uids),
        dtype=torch.long,
    )
    train_idx = torch.arange(0, num_train, dtype=torch.long)
    val_idx = torch.arange(num_train, num_train + num_val, dtype=torch.long)
    test_idx = torch.arange(num_train + num_val, num_train + num_val + num_test, dtype=torch.long)

    print("[4/6] Encoding descriptions and tweets...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device).eval()
    des_tensor = encode_descriptions(
        all_uids,
        profiles,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )
    tweets_tensor = encode_tweets(
        all_uids,
        tweets_by_uid,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )
    del model

    print("[5/6] Saving tensors and metadata...")
    torch.save(labels_tensor.cpu(), output_dir / "label.pt")
    torch.save(train_idx.cpu(), output_dir / "train_idx.pt")
    torch.save(val_idx.cpu(), output_dir / "val_idx.pt")
    torch.save(test_idx.cpu(), output_dir / "test_idx.pt")
    torch.save(edge_index.cpu(), output_dir / "edge_index.pt")
    torch.save(edge_type.cpu(), output_dir / "edge_type.pt")
    torch.save(num_properties_tensor.cpu(), output_dir / "num_properties_tensor.pt")
    torch.save(cat_properties_tensor.cpu(), output_dir / "cat_properties_tensor.pt")
    torch.save(des_tensor.cpu(), output_dir / "des_tensor.pt")
    torch.save(tweets_tensor.cpu(), output_dir / "tweets_tensor.pt")

    save_json(output_dir / "all_uids.json", all_uids)
    save_json(output_dir / "labeled_uids.json", labeled_uids)
    save_json(output_dir / "agent_uids.json", agent_uids)
    save_json(
        output_dir / "metadata.json",
        {
            **metadata,
            "split_source": str(SPLIT_PATH),
            "seed": args.seed,
            "reference_date": args.reference_date,
            "device": str(device),
            "max_tweets": args.max_tweets,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "num_feature_order": [
                "followers_count",
                "active_days",
                "screen_name_length",
                "friends_count",
                "interactions_count_proxy_listed_count",
            ],
            "cat_feature_order": ["default_profile_image"],
            "agent_label_value": AGENT_LABEL,
            "label_policy": {
                "0": "human_to_0",
                "1": "suspicious_drop",
                "2": "bot_to_1",
                "-1": "agent_keep_for_graph_use_label_2",
            },
            "num_feature_mean": mean.tolist(),
            "num_feature_std": std.tolist(),
            "edge_count": int(edge_index.shape[1]),
        },
    )

    print("[6/6] Done.")
    print(f"  label: {tuple(labels_tensor.shape)}")
    print(f"  edge_index: {tuple(edge_index.shape)}")
    print(f"  num_properties_tensor: {tuple(num_properties_tensor.shape)}")
    print(f"  cat_properties_tensor: {tuple(cat_properties_tensor.shape)}")
    print(f"  des_tensor: {tuple(des_tensor.shape)}")
    print(f"  tweets_tensor: {tuple(tweets_tensor.shape)}")


if __name__ == "__main__":
    main()