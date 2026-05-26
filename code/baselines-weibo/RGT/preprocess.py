"""
preprocess.py — RGT (Weibo) 预处理脚本
python preprocess.py [--output-dir PATH] [--seed 42] [--batch-size 32]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Weibo"
BERT_PATH = ROOT_DIR / "models" / "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = BASE_DIR / "processed"
SPLIT_PATH = DATA_DIR / "split.json"
REFERENCE_DATE = datetime(2026, 4, 1)
MAX_TWEETS = 20


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RGT Weibo Preprocessing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


# ── 设备检测 ──────────────────────────────────────────────────────────────────
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── I/O ───────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes"}
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def active_days(created_at_str):
    dt = parse_dt(created_at_str)
    if dt is None:
        return 1.0
    return max((REFERENCE_DATE - dt).days, 1)


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_profiles():
    profiles = {}
    for item in iter_jsonl(DATA_DIR / "user_profile.jsonl"):
        uid = str(item.get("uid", ""))
        if uid:
            profiles[uid] = item
    return profiles


def load_tweets():
    """返回 {uid: [text, ...]}（按 created_at 降序，最多 MAX_TWEETS 条非空文本）"""
    raw = {}
    for item in iter_jsonl(DATA_DIR / "user_tweets.jsonl"):
        uid = str(item.get("uid", ""))
        if not uid:
            continue
        text = (item.get("text") or "").strip()
        ts = parse_dt(item.get("created_at"))
        raw.setdefault(uid, []).append((ts, text))

    result = {}
    for uid, items in raw.items():
        items.sort(key=lambda x: x[0] or datetime.min, reverse=True)
        texts = [t for _, t in items if t][:MAX_TWEETS]
        result[uid] = texts
    return result


def load_followings():
    """返回 set of (source_uid, target_uid)"""
    edges = set()
    for item in iter_jsonl(DATA_DIR / "followings.jsonl"):
        src = str(item.get("source_uid", ""))
        tgt = str(item.get("target_uid", ""))
        if src and tgt:
            edges.add((src, tgt))
    return edges


def build_split_order(label_dict, split_dict, profiles):
    # 这里只为监督样本维护 train/val/test 三个桶；OOD(agent) 单独放在 agent_uids 里。
    split_buckets = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    agent_uids = []
    dropped_missing_split = []
    dropped_unexpected_ood = []

    for uid, raw_label in label_dict.items():
        if uid not in profiles:
            continue
        split_name = split_dict.get(uid)

        # label_dict 里的值是标签，不是 split：
        #   0=human, 2=bot, -1=agent(OOD), 1=疑似(直接丢弃)
        if raw_label == 0:
            mapped_label = 0
        elif raw_label == 2:
            mapped_label = 1
        elif raw_label == -1:
            # agent 节点不参与监督训练；如果官方 split 提供了 ood，就与这里保持一致。
            if split_name not in {None, "ood"}:
                dropped_missing_split.append(uid)
                continue
            agent_uids.append(uid)
            continue
        else:
            continue

        if split_name == "ood":
            dropped_unexpected_ood.append(uid)
            continue
        if split_name not in split_buckets:
            dropped_missing_split.append(uid)
            continue
        split_buckets[split_name].append(uid)
        split_labels[split_name].append(mapped_label)

    ordered_labeled = (
        split_buckets["train"] +
        split_buckets["val"] +
        split_buckets["test"]
    )
    ordered_labels = (
        split_labels["train"] +
        split_labels["val"] +
        split_labels["test"]
    )
    metadata = {
        "n_train": len(split_buckets["train"]),
        "n_val": len(split_buckets["val"]),
        "n_test": len(split_buckets["test"]),
        "n_agent": len(agent_uids),
        "dropped_missing_split_count": len(dropped_missing_split),
        "dropped_missing_split_uids": dropped_missing_split,
        "dropped_unexpected_ood_count": len(dropped_unexpected_ood),
        "dropped_unexpected_ood_uids": dropped_unexpected_ood,
    }
    return ordered_labeled, ordered_labels, agent_uids, metadata


# ── 特征提取 ──────────────────────────────────────────────────────────────────
def build_cat_features(uids, profiles):
    """[N, 2] — verified, default_profile_image"""
    rows = []
    for uid in uids:
        p = profiles.get(uid, {})
        rows.append([
            float(safe_bool(p.get("verified"))),
            float(safe_bool(p.get("default_profile_image"))),
        ])
    return np.asarray(rows, dtype=np.float32)


def build_num_features(uids, profiles):
    """[N, 5] — followers_count, active_days, name_len, friends_count, interactions_count"""
    rows = []
    for uid in uids:
        p = profiles.get(uid, {})
        followers   = float(safe_int(p.get("followers_count")))
        act_days    = float(active_days(p.get("created_at")))
        name_len    = float(len((p.get("screen_name") or "").strip()))
        friends     = float(safe_int(p.get("friends_count")))
        interactions = float(safe_int(p.get("interactions_count")))
        rows.append([followers, act_days, name_len, friends, interactions])
    return np.asarray(rows, dtype=np.float32)


def zscore_normalize(train_arr, arrays):
    """以训练集 mean/std 对所有切片做 z-score 归一化"""
    mean = train_arr.mean(axis=0)
    std  = train_arr.std(axis=0)
    std[std == 0] = 1.0
    return [(arr - mean) / std for arr in arrays], mean, std


# ── BERT 编码 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size, max_length):
    """
    texts: List[str]，空字符串 → zeros(768)
    返回 CPU Tensor [N, 768]
    """
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
        batch_texts = texts[i: i + batch_size]
        # 空文本占位符（不能传空串给 tokenizer）
        non_empty = [(j, t) for j, t in enumerate(batch_texts) if t.strip()]
        batch_vecs = torch.zeros(len(batch_texts), 768)

        if non_empty:
            batch_idx, batch_strs = zip(*non_empty)
            enc = tokenizer(
                list(batch_strs),
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # 使用 pooler_output（[CLS] 经过线性层+tanh）
            if out.pooler_output is not None:
                emb = out.pooler_output.cpu()  # [B', 768]
            else:
                emb = out.last_hidden_state[:, 0, :].cpu()
            for local_j, global_j in enumerate(batch_idx):
                batch_vecs[global_j] = emb[local_j]

        vecs.append(batch_vecs)

    return torch.cat(vecs, dim=0)  # [N, 768], CPU


def build_des_tensor(uids, profiles, tokenizer, model, device, batch_size, max_length):
    texts = [(profiles.get(uid, {}).get("description") or "").strip() for uid in uids]
    return encode_texts(texts, tokenizer, model, device, batch_size, max_length)


def build_tweets_tensor(uids, tweets_by_uid, tokenizer, model, device, batch_size, max_length):
    """每用户用最多 MAX_TWEETS 条推文的均值向量，无推文 → zeros"""
    all_user_texts = {}
    unique_texts = []
    text_to_idx = {}

    for uid in uids:
        txts = tweets_by_uid.get(uid, [])
        all_user_texts[uid] = txts
        for t in txts:
            if t not in text_to_idx:
                text_to_idx[t] = len(unique_texts)
                unique_texts.append(t)

    # 编码所有唯一推文
    if unique_texts:
        all_embs = encode_texts(unique_texts, tokenizer, model, device, batch_size, max_length)
    else:
        all_embs = torch.zeros(0, 768)

    # 每用户取均值
    user_vecs = []
    for uid in uids:
        txts = all_user_texts.get(uid, [])
        if not txts:
            user_vecs.append(torch.zeros(768))
        else:
            indices = [text_to_idx[t] for t in txts]
            user_vecs.append(all_embs[indices].mean(0))

    return torch.stack(user_vecs, dim=0)  # [N, 768], CPU


# ── 边构建 ────────────────────────────────────────────────────────────────────
def build_edges(uid_to_index, raw_edges):
    """
    raw_edges: set of (source_uid, target_uid)
    edge_type 0 = 互相关注（A→B 且 B→A）
    edge_type 1 = 单向关注（A→B 仅）
    过滤：两端都必须在 uid_to_index 中
    """
    mutual_pairs = []
    one_way_pairs = []

    for (src, tgt) in raw_edges:
        if src not in uid_to_index or tgt not in uid_to_index:
            continue
        src_i = uid_to_index[src]
        tgt_i = uid_to_index[tgt]
        if src_i == tgt_i:
            continue
        if (tgt, src) in raw_edges:
            # 互关：只记录 src < tgt 的方向，避免重复
            if src_i < tgt_i:
                mutual_pairs.append((src_i, tgt_i))
                mutual_pairs.append((tgt_i, src_i))
        else:
            one_way_pairs.append((src_i, tgt_i))

    if mutual_pairs or one_way_pairs:
        all_pairs = mutual_pairs + one_way_pairs
        edge_index = torch.tensor(all_pairs, dtype=torch.long).t().contiguous()
        edge_type  = torch.cat([
            torch.zeros(len(mutual_pairs), dtype=torch.long),
            torch.ones(len(one_way_pairs),  dtype=torch.long),
        ])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type  = torch.empty((0,),   dtype=torch.long)

    return edge_index, edge_type


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")

    # ── 1. 加载标签 ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading labels, profiles, and official split...")
    label_dict = {str(k): int(v) for k, v in load_json(DATA_DIR / "Weibo_Labels.json").items()}
    profiles = load_profiles()
    split_dict = {str(uid): split for uid, split in load_json(SPLIT_PATH).items()}

    ordered_labeled, ordered_labels, agent_uids, split_meta = build_split_order(
        label_dict,
        split_dict,
        profiles,
    )

    print(
        f"  labeled: {len(ordered_labeled)}  "
        f"(train={split_meta['n_train']}, val={split_meta['n_val']}, test={split_meta['n_test']})"
    )
    print(f"  agent:   {len(agent_uids)}")
    if split_meta["dropped_missing_split_count"]:
        print(f"  dropped(no official split): {split_meta['dropped_missing_split_count']}")
    if split_meta["dropped_unexpected_ood_count"]:
        print(f"  dropped(supervised label but split=ood): {split_meta['dropped_unexpected_ood_count']}")

    # ── 2. 按官方 split 重排 train/val/test ─────────────────────────────────
    print("\n[2/7] Reordering nodes by official split.json...")

    # all_uids 顺序：train → val → test → agent
    all_uids = ordered_labeled + agent_uids
    uid_to_index = {uid: i for i, uid in enumerate(all_uids)}

    n_train = split_meta["n_train"]
    n_val   = split_meta["n_val"]
    n_test  = split_meta["n_test"]
    n_agent = len(agent_uids)
    n_total = len(all_uids)

    train_idx = torch.arange(0, n_train, dtype=torch.long)
    val_idx   = torch.arange(n_train, n_train + n_val, dtype=torch.long)
    test_idx  = torch.arange(n_train + n_val, n_train + n_val + n_test, dtype=torch.long)
    agent_idx = torch.arange(n_train + n_val + n_test, n_total, dtype=torch.long)

    # ── 3. 分类特征 ──────────────────────────────────────────────────────────
    print("\n[3/7] Building categorical features [N, 2]...")
    cat_arr = build_cat_features(all_uids, profiles)
    cat_tensor = torch.tensor(cat_arr, dtype=torch.float32)
    print(f"  shape: {cat_tensor.shape}")

    # ── 4. 数值特征 ──────────────────────────────────────────────────────────
    print("\n[4/7] Building numerical features [N, 5]...")
    num_arr_all = build_num_features(all_uids, profiles)

    # 用训练集做 z-score（agent 也用 train 的 mean/std 归一化）
    train_num = num_arr_all[:n_train]
    [num_arr_norm], num_mean, num_std = zscore_normalize(train_num, [num_arr_all])
    num_tensor = torch.tensor(num_arr_norm, dtype=torch.float32)
    print(f"  shape: {num_tensor.shape}")

    # ── 5. BERT 编码 ─────────────────────────────────────────────────────────
    print(f"\n[5/7] Loading BERT model from {BERT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_PATH))
    bert_model = AutoModel.from_pretrained(str(BERT_PATH)).to(device).eval()

    print("  Encoding descriptions...")
    des_tensor = build_des_tensor(
        all_uids, profiles, tokenizer, bert_model, device,
        args.batch_size, args.max_length
    )
    print(f"  des_tensor shape: {des_tensor.shape}")

    print("  Loading tweets...")
    tweets_by_uid = load_tweets()
    print(f"  Users with tweets: {len(tweets_by_uid)}")

    print("  Encoding tweets...")
    tweets_tensor = build_tweets_tensor(
        all_uids, tweets_by_uid, tokenizer, bert_model, device,
        args.batch_size, args.max_length
    )
    print(f"  tweets_tensor shape: {tweets_tensor.shape}")

    del bert_model  # 释放显存

    # ── 6. 构建图边 ──────────────────────────────────────────────────────────
    print("\n[6/7] Building graph edges...")
    raw_edges = load_followings()
    print(f"  Raw following edges: {len(raw_edges)}")
    edge_index, edge_type = build_edges(uid_to_index, raw_edges)
    mutual_count  = int((edge_type == 0).sum())
    one_way_count = int((edge_type == 1).sum())
    print(f"  Filtered edges: {edge_index.shape[1]}  (mutual={mutual_count}, one-way={one_way_count})")

    # ── 7. 构造标签 tensor ───────────────────────────────────────────────────
    print("\n[7/7] Building label tensor...")
    label_list = ordered_labels + [2] * n_agent  # 2 = agent (OOD)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    print(f"  label shape: {label_tensor.shape}  (0={label_list.count(0)}, 1={label_list.count(1)}, 2={label_list.count(2)})")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    print("\nSaving tensors to", args.output_dir, "...")

    def save(tensor, name):
        torch.save(tensor.cpu(), args.output_dir / name)
        print(f"  Saved {name}  {tuple(tensor.shape)}")

    save(cat_tensor,    "cat_properties_tensor.pt")
    save(num_tensor,    "num_properties_tensor.pt")
    save(des_tensor,    "des_tensor.pt")
    save(tweets_tensor, "tweets_tensor.pt")
    save(edge_index,    "edge_index.pt")
    save(edge_type,     "edge_type.pt")
    save(label_tensor,  "label.pt")
    save(train_idx,     "train_idx.pt")
    save(val_idx,       "val_idx.pt")
    save(test_idx,      "test_idx.pt")
    save(agent_idx,     "agent_idx.pt")

    with open(args.output_dir / "all_uids.json", "w", encoding="utf-8") as f:
        json.dump(all_uids, f, ensure_ascii=False, indent=2)

    metadata = {
        "n_total": n_total, "n_train": n_train, "n_val": n_val,
        "n_test": n_test, "n_agent": n_agent,
        "num_edges": int(edge_index.shape[1]),
        "mutual_edges": mutual_count, "one_way_edges": one_way_count,
        "label_counts": {"human": label_list.count(0), "bot": label_list.count(1), "agent": label_list.count(2)},
        "seed": args.seed,
        "split_source": str(SPLIT_PATH),
        "dropped_missing_split_count": split_meta["dropped_missing_split_count"],
        "dropped_unexpected_ood_count": split_meta["dropped_unexpected_ood_count"],
        "reference_date": REFERENCE_DATE.strftime("%Y-%m-%d"),
        "num_feature_order": [
            "followers_count",
            "active_days",
            "screen_name_length",
            "friends_count",
            "interactions_count",
        ],
        "num_feature_mean": num_mean.tolist(),
        "num_feature_std": num_std.tolist(),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nPreprocessing complete.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
