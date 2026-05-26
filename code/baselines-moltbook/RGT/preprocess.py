"""
  cd baselines-moltbook/RGT
  python preprocess.py
  python preprocess.py --output-dir ./processed --weibo-rgt-processed-dir ../../baselines-weibo/RGT/processed
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
DATA_DIR = ROOT_DIR / "dataset" / "Moltbook"
BERT_PATH = ROOT_DIR / "models" / "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = BASE_DIR / "processed"
DEFAULT_WEIBO_RGT_PROCESSED = ROOT_DIR / "baselines-weibo" / "RGT" / "processed"
REFERENCE_DATE = datetime(2026, 4, 1)
MAX_TWEETS = 20


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RGT Moltbook OOD Preprocessing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"预处理结果输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--weibo-rgt-processed-dir",
        type=Path,
        default=DEFAULT_WEIBO_RGT_PROCESSED,
        help=f"Weibo RGT processed 目录（含 metadata.json，用于读取归一化统计量）\n"
             f"（默认：{DEFAULT_WEIBO_RGT_PROCESSED}）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备：auto / cuda / mps / cpu（默认：auto）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="BERT 编码批大小（默认：32）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="BERT tokenizer 最大序列长度（默认：128）",
    )
    parser.add_argument(
        "--max-tweets",
        type=int,
        default=MAX_TWEETS,
        help=f"每用户最多使用的历史帖文数量（默认：{MAX_TWEETS}）",
    )
    return parser.parse_args()


# ── 设备检测 ──────────────────────────────────────────────────────────────────
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


# ── I/O ───────────────────────────────────────────────────────────────────────
def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def parse_join_date(value: str):
    """解析 join_time 字段（格式 "%Y-%m-%d"），返回 datetime 或 None。"""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None


# ── Step 1: 加载用户信息 ──────────────────────────────────────────────────────
def load_user_info():
    """
    加载 user_info.jsonl，以 author 为 UID。
    返回：
        profiles: {author: dict}
        all_uids: [author, ...]（保留文件顺序，去重）
    """
    profiles = {}
    all_uids = []
    for item in iter_jsonl(DATA_DIR / "user_info.jsonl"):
        author = (item.get("author") or "").strip()
        if not author or author in profiles:
            continue
        profiles[author] = item
        all_uids.append(author)
    return profiles, all_uids


# ── Step 2: 构建回复关系图 ───────────────────────────────────────────────────
def build_reply_edges(all_uids: list, profiles: dict):
    """
    通过 post_records.jsonl 和 comment_records.jsonl 构建回复关系边。

    流程：
      1. post_records: url -> author（谁发了该帖）
      2. comment_records: 评论的 post_url 对应的 author 即为被回复者
         => 有向边：(comment.author → post.author)
      3. 双方均须在 profiles 中；去重（只有无有，不考虑回复次数）；过滤自环

    边类型：
      type 0 = 互相回复（A→B 且 B→A 均存在）
      type 1 = 单向回复（仅 A→B 存在）

    返回：
        edge_index: [2, E] long tensor
        edge_type:  [E]    long tensor，0=互相回复，1=单向回复
    """
    uid_to_index = {uid: idx for idx, uid in enumerate(all_uids)}

    # 建立 post_url → post_author 映射
    post_url_to_author = {}
    for item in iter_jsonl(DATA_DIR / "post_records.jsonl"):
        url = (item.get("url") or "").strip()
        author = (item.get("author") or "").strip()
        if url and author:
            post_url_to_author[url] = author

    # 从评论生成原始有向边集合（去重后）
    seen = set()
    directed_edges = set()  # set of (src_idx, tgt_idx)
    for item in iter_jsonl(DATA_DIR / "comment_records.jsonl"):
        commenter = (item.get("author") or "").strip()
        post_url = (item.get("post_url") or "").strip()
        if not commenter or not post_url:
            continue
        post_author = post_url_to_author.get(post_url, "")
        if not post_author:
            continue
        # 双方须在 user_info 中
        if commenter not in uid_to_index or post_author not in uid_to_index:
            continue
        src_idx = uid_to_index[commenter]
        tgt_idx = uid_to_index[post_author]
        # 过滤自环
        if src_idx == tgt_idx:
            continue
        directed_edges.add((src_idx, tgt_idx))

    # 区分互相回复（type 0）和单向回复（type 1）
    mutual_pairs = []
    one_way_pairs = []
    already_mutual = set()

    for (src, tgt) in directed_edges:
        if src < tgt and (tgt, src) in directed_edges:
            # 互相回复：两个方向都记录，type 0
            mutual_pairs.append((src, tgt))
            mutual_pairs.append((tgt, src))
            already_mutual.add((src, tgt))
            already_mutual.add((tgt, src))

    for (src, tgt) in directed_edges:
        if (src, tgt) not in already_mutual:
            one_way_pairs.append((src, tgt))

    if mutual_pairs or one_way_pairs:
        all_pairs = mutual_pairs + one_way_pairs
        edge_index = torch.tensor(all_pairs, dtype=torch.long).t().contiguous()
        edge_type = torch.cat([
            torch.zeros(len(mutual_pairs), dtype=torch.long),
            torch.ones(len(one_way_pairs), dtype=torch.long),
        ])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return edge_index, edge_type


# ── Step 3: 数值特征 ──────────────────────────────────────────────────────────
def build_num_features(all_uids: list, profiles: dict) -> np.ndarray:
    """
    构建 [N, 5] 数值特征矩阵（未归一化），特征顺序与 Weibo RGT 完全对齐：
      [0] followers_count    <- profiles[uid]["followers"]
      [1] active_days        <- (REFERENCE_DATE - join_time).days
      [2] screen_name_length <- len(author)
      [3] friends_count      <- profiles[uid]["following"]
      [4] interactions_count <- profiles[uid]["karma"]（用户确认）
    """
    rows = []
    for uid in all_uids:
        p = profiles[uid]
        join_dt = parse_join_date(p.get("join_time"))
        if join_dt is not None:
            active_days = max((REFERENCE_DATE - join_dt).days, 1)
        else:
            active_days = 1

        rows.append([
            float(safe_int(p.get("followers"))),
            float(active_days),
            float(len(uid)),             # uid == author name
            float(safe_int(p.get("following"))),
            float(safe_int(p.get("karma"))),
        ])
    return np.asarray(rows, dtype=np.float32)


def apply_weibo_normalization(
    raw_array: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """用 Weibo RGT 训练集统计量做 z-score 归一化。"""
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return ((raw_array - mean) / std_safe).astype(np.float32)


def load_weibo_rgt_norm_stats(weibo_rgt_processed_dir: Path):
    """
    从 baselines-weibo/RGT/processed/metadata.json 读取归一化统计量。
    若文件不存在，给出明确错误提示。
    """
    meta_path = weibo_rgt_processed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"找不到 Weibo RGT metadata.json: {meta_path}\n"
            "请先在 baselines-weibo/RGT/ 目录下运行:\n"
            "  python preprocess.py\n"
            "确保 processed/metadata.json 中包含 num_feature_mean 和 num_feature_std 后再运行本脚本。"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "num_feature_mean" not in meta or "num_feature_std" not in meta:
        raise KeyError(
            "metadata.json 中缺少 num_feature_mean / num_feature_std 字段。\n"
            "请在 baselines-weibo/RGT/ 目录下重新运行 preprocess.py 以生成这些字段。"
        )

    mean = np.asarray(meta["num_feature_mean"], dtype=np.float32)
    std = np.asarray(meta["num_feature_std"], dtype=np.float32)
    feature_order = meta.get("num_feature_order", [])
    return mean, std, feature_order


# ── Step 4: 类别特征 ──────────────────────────────────────────────────────────
def build_cat_features(all_uids: list, profiles: dict) -> np.ndarray:
    """
    类别特征 [N, 2]，与 Weibo RGT 对齐：
      [0] verified              <- user_info.verified（布尔）
      [1] default_profile_image <- 全部填 1（Moltbook 账号均使用默认头像）
    """
    rows = []
    for uid in all_uids:
        p = profiles[uid]
        verified = float(safe_bool(p.get("verified")))
        rows.append([verified, 1.0])
    return np.asarray(rows, dtype=np.float32)


# ── Step 5: 文本编码 ──────────────────────────────────────────────────────────
def get_sentence_embedding(model_outputs):
    """取 pooler_output；若不存在则取 last_hidden_state 的 [CLS] token。"""
    if getattr(model_outputs, "pooler_output", None) is not None:
        return model_outputs.pooler_output
    return model_outputs.last_hidden_state[:, 0, :]


@torch.no_grad()
def encode_texts(
    texts: list,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    progress_label: str,
) -> torch.Tensor:
    """批量编码文本列表，返回 [len(texts), hidden_size]。"""
    hidden_size = model.config.hidden_size
    if not texts:
        return torch.empty((0, hidden_size), dtype=torch.float32)

    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=progress_label):
        batch_texts = texts[start: start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings.append(get_sentence_embedding(outputs).detach().cpu())
    return torch.cat(embeddings, dim=0)


def encode_descriptions(
    all_uids: list,
    profiles: dict,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    编码 discription 字段（注意 Moltbook 数据中的拼写为 discription）。
    返回 [N, hidden_size]，无 description 的用户对应行全为 0。
    """
    descriptions = [(profiles[uid].get("discription") or "").strip() for uid in all_uids]
    hidden_size = model.config.hidden_size
    desc_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)

    non_empty_indices = [i for i, text in enumerate(descriptions) if text]
    non_empty_texts = [descriptions[i] for i in non_empty_indices]
    if non_empty_texts:
        encoded = encode_texts(
            non_empty_texts, tokenizer, model, device, batch_size, max_length,
            "Encoding descriptions"
        )
        desc_tensor[non_empty_indices] = encoded
    return desc_tensor


def load_history_posts(max_tweets: int) -> dict:
    """
    加载 user_history_posts.jsonl，返回 {author: [short_content, ...]}。
    按 time 降序排序，每用户最多保留 max_tweets 条非空文本。
    """
    raw = {}
    for item in iter_jsonl(DATA_DIR / "user_history_posts.jsonl"):
        author = (item.get("author") or "").strip()
        if not author:
            continue
        text = (item.get("short_content") or "").strip()
        time_str = item.get("time") or ""
        try:
            ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts = datetime.min
        raw.setdefault(author, []).append((ts, text))

    result = {}
    for author, items in raw.items():
        items.sort(key=lambda x: x[0], reverse=True)
        texts = [t for _, t in items if t][:max_tweets]
        result[author] = texts
    return result


def encode_tweets(
    all_uids: list,
    posts_by_author: dict,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    编码历史帖文（short_content），对每个用户的多条帖文取均值池化。
    返回 [N, hidden_size]，无帖文的用户对应行全为 0。
    """
    hidden_size = model.config.hidden_size
    tweet_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)

    flat_texts = []
    owners = []
    for node_idx, uid in enumerate(all_uids):
        for text in posts_by_author.get(uid, []):
            if text:
                flat_texts.append(text)
                owners.append(node_idx)

    if not flat_texts:
        return tweet_tensor

    encoded = encode_texts(
        flat_texts, tokenizer, model, device, batch_size, max_length,
        "Encoding history posts"
    )

    counts = torch.zeros(len(all_uids), dtype=torch.float32)
    for emb_idx, node_idx in enumerate(owners):
        tweet_tensor[node_idx] += encoded[emb_idx]
        counts[node_idx] += 1

    nonzero = counts > 0
    tweet_tensor[nonzero] = tweet_tensor[nonzero] / counts[nonzero].unsqueeze(1)
    return tweet_tensor


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    weibo_rgt_dir: Path = args.weibo_rgt_processed_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device(args.device)

    print("=== RGT Moltbook OOD Preprocessing ===")
    print(f"Data dir              : {DATA_DIR}")
    print(f"Output dir            : {output_dir}")
    print(f"Weibo RGT processed   : {weibo_rgt_dir}")
    print(f"Reference date        : {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    print(f"Device                : {device}")

    # ── [1/6] 加载用户信息 ────────────────────────────────────────────────────
    print("\n[1/6] Loading user info...")
    profiles, all_uids = load_user_info()
    N = len(all_uids)
    print(f"  Total agents: {N}")

    # ── [2/6] 构建回复关系图 ──────────────────────────────────────────────────
    print("\n[2/6] Building reply graph...")
    edge_index, edge_type = build_reply_edges(all_uids, profiles)
    mutual_count  = int((edge_type == 0).sum())
    one_way_count = int((edge_type == 1).sum())
    print(f"  Edges total: {edge_index.shape[1]}  (mutual={mutual_count}, one-way={one_way_count})")

    # ── [3/6] 构建特征并用 Weibo RGT 统计量归一化 ────────────────────────────
    print("\n[3/6] Building features and normalizing with Weibo RGT training stats...")
    mean, std, feature_order = load_weibo_rgt_norm_stats(weibo_rgt_dir)
    print(f"  Feature order : {feature_order}")
    print(f"  Weibo mean    : {mean.tolist()}")
    print(f"  Weibo std     : {std.tolist()}")

    raw_num = build_num_features(all_uids, profiles)        # [N, 5]
    norm_num = apply_weibo_normalization(raw_num, mean, std) # [N, 5]
    num_properties_tensor = torch.tensor(norm_num, dtype=torch.float32)

    raw_cat = build_cat_features(all_uids, profiles)        # [N, 2]
    cat_properties_tensor = torch.tensor(raw_cat, dtype=torch.float32)

    # 标签：全为 1（机器人）
    label_tensor = torch.ones(N, dtype=torch.long)
    # 无 train/val/test 划分
    empty_idx = torch.empty((0,), dtype=torch.long)

    # ── [4/6] BERT 文本编码 ───────────────────────────────────────────────────
    print(f"\n[4/6] Loading BERT model from {BERT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_PATH))
    bert_model = AutoModel.from_pretrained(str(BERT_PATH)).to(device).eval()

    print("  Encoding descriptions (discription field)...")
    des_tensor = encode_descriptions(
        all_uids, profiles, tokenizer, bert_model, device,
        args.batch_size, args.max_length
    )
    print(f"  des_tensor shape: {des_tensor.shape}")

    print("  Loading history posts...")
    posts_by_author = load_history_posts(args.max_tweets)
    print(f"  Users with history posts: {len(posts_by_author)}")

    print("  Encoding history posts...")
    tweets_tensor = encode_tweets(
        all_uids, posts_by_author, tokenizer, bert_model, device,
        args.batch_size, args.max_length
    )
    print(f"  tweets_tensor shape: {tweets_tensor.shape}")

    del bert_model  # 释放内存

    # ── [5/6] 保存张量文件 ────────────────────────────────────────────────────
    print("\n[5/6] Saving tensors...")

    def save(tensor, name):
        torch.save(tensor.cpu(), output_dir / name)
        print(f"  Saved {name}  {tuple(tensor.shape)}")

    save(cat_properties_tensor, "cat_properties_tensor.pt")
    save(num_properties_tensor, "num_properties_tensor.pt")
    save(des_tensor,            "des_tensor.pt")
    save(tweets_tensor,         "tweets_tensor.pt")
    save(edge_index,            "edge_index.pt")
    save(edge_type,             "edge_type.pt")
    save(label_tensor,          "label.pt")
    save(empty_idx,             "train_idx.pt")
    save(empty_idx,             "val_idx.pt")
    save(empty_idx,             "test_idx.pt")

    save_json(output_dir / "all_uids.json", all_uids)

    # ── [6/6] 保存 metadata ───────────────────────────────────────────────────
    posts_count = sum(len(v) for v in posts_by_author.values())
    metadata = {
        "dataset": "Moltbook",
        "purpose": "OOD evaluation — all users are agents (bots), ground truth = 1",
        "num_total_nodes": N,
        "num_train": 0,
        "num_val": 0,
        "num_test": 0,
        "edge_count": int(edge_index.shape[1]),
        "mutual_edges": mutual_count,
        "one_way_edges": one_way_count,
        "reference_date": REFERENCE_DATE.strftime("%Y-%m-%d"),
        "device": str(device),
        "max_tweets": args.max_tweets,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_feature_order": [
            "followers_count",
            "active_days",
            "screen_name_length",
            "friends_count",
            "interactions_count_proxy_karma",
        ],
        "num_feature_mapping": {
            "followers_count": "user_info.followers",
            "active_days": "join_time ('%Y-%m-%d') to reference_date",
            "screen_name_length": "len(author)",
            "friends_count": "user_info.following",
            "interactions_count_proxy": "user_info.karma (confirmed by user)",
        },
        "cat_feature_order": ["verified", "default_profile_image"],
        "cat_feature_mapping": {
            "verified": "user_info.verified (bool)",
            "default_profile_image": "all ones (Moltbook accounts all use default avatar)",
        },
        "graph_construction": {
            "method": "reply_graph",
            "source": "comment_records.jsonl + post_records.jsonl",
            "edge_type_0": "mutual reply (A replied B's post AND B replied A's post)",
            "edge_type_1": "one-way reply (A replied B's post, but B did NOT reply A's post)",
            "rule": "deduplicated; no self-loops; both nodes must be in user_info",
        },
        "normalization": {
            "method": "z-score using Weibo RGT training set statistics",
            "stats_source": str(weibo_rgt_dir / "metadata.json"),
            "num_feature_mean": mean.tolist(),
            "num_feature_std": std.tolist(),
        },
        "text_encoding": {
            "model": str(BERT_PATH),
            "description_field": "discription (note: Moltbook typo)",
            "post_field": "user_history_posts.jsonl short_content (mean pooling, top-{} by time)".format(
                args.max_tweets
            ),
        },
        "label_info": {
            "all_labels": 1,
            "meaning": "all agents are bots (OOD ground truth)",
        },
        "posts_with_text_count": posts_count,
    }
    save_json(output_dir / "metadata.json", metadata)

    print("\n[6/6] Done.")
    print(f"  cat_properties_tensor : {tuple(cat_properties_tensor.shape)}")
    print(f"  num_properties_tensor : {tuple(num_properties_tensor.shape)}")
    print(f"  des_tensor            : {tuple(des_tensor.shape)}")
    print(f"  tweets_tensor         : {tuple(tweets_tensor.shape)}")
    print(f"  edge_index            : {tuple(edge_index.shape)}")
    print(f"  label                 : {tuple(label_tensor.shape)}")
    print(f"  Output dir            : {output_dir}")


if __name__ == "__main__":
    main()
