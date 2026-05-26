"""
  cd baselines-moltbook/moltbook-preprocess
  python preprocess_moltbook.py
  python preprocess_moltbook.py --output-dir ./moltbook_processed_data \\
      --weibo-processed-dir ../../baselines-weibo/weibo-preprocess/weibo_processed_data
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
MODEL_DIR = ROOT_DIR / "models" / "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = BASE_DIR / "moltbook_processed_data"
DEFAULT_WEIBO_PROCESSED_DIR = (
    ROOT_DIR / "baselines-weibo" / "weibo-preprocess" / "weibo_processed_data"
)
REFERENCE_DATE = "2026-04-01"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Moltbook data into the weibo-preprocess-compatible format for OOD GNN evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"预处理结果输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--weibo-processed-dir",
        type=Path,
        default=DEFAULT_WEIBO_PROCESSED_DIR,
        help="weibo-preprocess 输出目录，用于加载归一化统计量（默认："
        f"{DEFAULT_WEIBO_PROCESSED_DIR}）",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=REFERENCE_DATE,
        help=f"参考日期，用于计算账户年限（默认：{REFERENCE_DATE}）",
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
        default=20,
        help="每个用户最多使用的历史帖文数量（默认：20）",
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
def iterate_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_int(value, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_join_date(value: str):
    """解析 join_time 字段（格式 '%Y-%m-%d'），返回 datetime 对象或 None。"""
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
        all_uids: [author, ...]（保留文件顺序）
    """
    profiles = {}
    all_uids = []
    for item in iterate_jsonl(DATA_DIR / "user_info.jsonl"):
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
         => 有向边：(comment.author, post.author)
      3. 双方均须在 profiles 中，且去重（只有无有，不考虑回复次数）
      4. 过滤自环

    返回：
        edge_pairs: [(src_idx, tgt_idx), ...]
    """
    uid_to_index = {uid: idx for idx, uid in enumerate(all_uids)}

    # 建立 post_url → post_author 映射
    post_url_to_author = {}
    for item in iterate_jsonl(DATA_DIR / "post_records.jsonl"):
        url = (item.get("url") or "").strip()
        author = (item.get("author") or "").strip()
        if url and author:
            post_url_to_author[url] = author

    # 从评论生成边
    seen = set()
    edge_pairs = []
    for item in iterate_jsonl(DATA_DIR / "comment_records.jsonl"):
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
        pair = (src_idx, tgt_idx)
        if pair in seen:
            continue
        seen.add(pair)
        edge_pairs.append(pair)

    return edge_pairs


def build_edge_tensors(edge_pairs):
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_type = torch.zeros(len(edge_pairs), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    return edge_index, edge_type


# ── Step 3: 数值特征 ──────────────────────────────────────────────────────────
def build_num_features(all_uids: list, profiles: dict, reference_time: datetime) -> np.ndarray:
    """
    构建 [N, 5] 数值特征矩阵（未归一化）。
    特征顺序（与 weibo-preprocess 完全对齐）：
      [0] followers_count    <- profiles[uid]["followers"]
      [1] active_days        <- (reference_time - join_time).days
      [2] screen_name_length <- len(author)
      [3] friends_count      <- profiles[uid]["following"]
      [4] interactions_count <- profiles[uid]["karma"]  (用户确认)
    """
    rows = []
    for uid in all_uids:
        profile = profiles[uid]
        join_dt = parse_join_date(profile.get("join_time"))
        if join_dt is not None:
            active_days = max((reference_time - join_dt).days, 1)
        else:
            active_days = 1

        rows.append(
            [
                float(safe_int(profile.get("followers"))),
                float(active_days),
                float(len(uid)),              # uid == author name
                float(safe_int(profile.get("following"))),
                float(safe_int(profile.get("karma"))),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def apply_weibo_normalization(raw_array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """用 Weibo 训练集统计量做 z-score 归一化。"""
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return ((raw_array - mean) / std_safe).astype(np.float32)


def load_weibo_norm_stats(weibo_processed_dir: Path):
    """
    从 weibo_processed_data/metadata.json 读取归一化统计量。
    若目录或文件不存在，给出明确错误提示。
    """
    meta_path = weibo_processed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"找不到 Weibo 预处理 metadata.json: {meta_path}\n"
            "请先在 baselines-weibo/weibo-preprocess/ 目录下运行:\n"
            "  python preprocess_weibo.py\n"
            "确保 weibo_processed_data/metadata.json 存在后再运行本脚本。"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mean = np.asarray(meta["num_feature_mean"], dtype=np.float32)
    std = np.asarray(meta["num_feature_std"], dtype=np.float32)
    feature_order = meta.get("num_feature_order", [])
    return mean, std, feature_order


# ── Step 4: 类别特征 ──────────────────────────────────────────────────────────
def build_cat_features(n: int) -> np.ndarray:
    """
    类别特征 [N, 1]，全为 1。
    对应 Weibo 的 default_profile_image 字段，
    Moltbook 账号均使用默认头像，故全部填 1（是默认头像）。
    """
    return np.ones((n, 1), dtype=np.float32)


# ── Step 5: 文本编码 ──────────────────────────────────────────────────────────
def get_sentence_embedding(model_outputs):
    """取 pooler_output，若不存在则取 [CLS] token 的 last_hidden_state。"""
    if getattr(model_outputs, "pooler_output", None) is not None:
        return model_outputs.pooler_output
    return model_outputs.last_hidden_state[:, 0, :]


@torch.no_grad()
def encode_texts(texts: list, tokenizer, model, device: torch.device, batch_size: int, max_length: int, progress_label: str) -> torch.Tensor:
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


def encode_descriptions(all_uids: list, profiles: dict, tokenizer, model, device, batch_size, max_length) -> torch.Tensor:
    """
    编码 discription 字段（注意 Moltbook 数据的拼写为 discription）。
    返回 [N, hidden_size]，无 description 的用户对应行全为 0。
    """
    # 注意：Moltbook 字段名拼写为 discription（有 typo）
    descriptions = [(profiles[uid].get("discription") or "").strip() for uid in all_uids]
    hidden_size = model.config.hidden_size
    desc_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)

    non_empty_indices = [i for i, text in enumerate(descriptions) if text]
    non_empty_texts = [descriptions[i] for i in non_empty_indices]
    if non_empty_texts:
        encoded = encode_texts(non_empty_texts, tokenizer, model, device, batch_size, max_length, "Encoding descriptions")
        desc_tensor[non_empty_indices] = encoded
    return desc_tensor


def load_history_posts(max_tweets: int) -> dict:
    """
    加载 user_history_posts.jsonl，返回 {author: [short_content, ...]}。
    按 time 降序排序，每用户最多保留 max_tweets 条非空文本。
    """
    raw = {}
    for item in iterate_jsonl(DATA_DIR / "user_history_posts.jsonl"):
        author = (item.get("author") or "").strip()
        if not author:
            continue
        text = (item.get("short_content") or "").strip()
        # time 字段格式为 "%Y-%m-%d %H:%M:%S"
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


def encode_tweets(all_uids: list, posts_by_author: dict, tokenizer, model, device, batch_size, max_length) -> torch.Tensor:
    """
    编码历史帖文（short_content），对每个用户的多条帖文取均值池化。
    返回 [N, hidden_size]，无帖文的用户对应行全为 0。
    """
    hidden_size = model.config.hidden_size
    tweet_tensor = torch.zeros((len(all_uids), hidden_size), dtype=torch.float32)

    flat_texts = []
    owners = []  # 每条文本对应的节点索引
    for node_idx, uid in enumerate(all_uids):
        for text in posts_by_author.get(uid, []):
            if text:
                flat_texts.append(text)
                owners.append(node_idx)

    if not flat_texts:
        return tweet_tensor

    encoded = encode_texts(flat_texts, tokenizer, model, device, batch_size, max_length, "Encoding posts")

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
    weibo_proc_dir: Path = args.weibo_processed_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_time = datetime.strptime(args.reference_date, "%Y-%m-%d")
    device = detect_device(args.device)

    print("=== Moltbook GNN Preprocessing (OOD) ===")
    print(f"Data dir         : {DATA_DIR}")
    print(f"Output dir       : {output_dir}")
    print(f"Weibo stats dir  : {weibo_proc_dir}")
    print(f"Reference date   : {args.reference_date}")
    print(f"Device           : {device}")

    # ── [1/6] 加载用户信息 ────────────────────────────────────────────────────
    print("\n[1/6] Loading user info...")
    profiles, all_uids = load_user_info()
    N = len(all_uids)
    print(f"  Total agents: {N}")

    # ── [2/6] 构建回复关系图 ──────────────────────────────────────────────────
    print("\n[2/6] Building reply graph...")
    edge_pairs = build_reply_edges(all_uids, profiles)
    edge_index, edge_type = build_edge_tensors(edge_pairs)
    print(f"  Edges (deduplicated): {edge_index.shape[1]}")

    # ── [3/6] 构建数值/类别特征，加载 Weibo 归一化统计量 ─────────────────────
    print("\n[3/6] Building features and normalizing with Weibo training stats...")
    mean, std, feature_order = load_weibo_norm_stats(weibo_proc_dir)
    print(f"  Weibo num_feature_order: {feature_order}")
    print(f"  Weibo mean: {mean.tolist()}")
    print(f"  Weibo std : {std.tolist()}")

    raw_num = build_num_features(all_uids, profiles, reference_time)   # [N, 5]
    norm_num = apply_weibo_normalization(raw_num, mean, std)            # [N, 5]

    num_properties_tensor = torch.tensor(norm_num, dtype=torch.float32)
    cat_properties_tensor = torch.tensor(build_cat_features(N), dtype=torch.float32)

    # 标签：全为 1（机器人），train/val/test 索引均为空
    labels_tensor = torch.ones(N, dtype=torch.long)
    empty_idx = torch.empty((0,), dtype=torch.long)

    # ── [4/6] BERT 文本编码 ───────────────────────────────────────────────────
    print("\n[4/6] Loading BERT model and encoding texts...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device).eval()

    des_tensor = encode_descriptions(
        all_uids, profiles, tokenizer, model, device, args.batch_size, args.max_length
    )
    posts_by_author = load_history_posts(args.max_tweets)
    tweets_tensor = encode_tweets(
        all_uids, posts_by_author, tokenizer, model, device, args.batch_size, args.max_length
    )
    del model

    # ── [5/6] 保存张量文件 ────────────────────────────────────────────────────
    print("\n[5/6] Saving tensors...")
    torch.save(labels_tensor.cpu(),           output_dir / "label.pt")
    torch.save(empty_idx.cpu(),               output_dir / "train_idx.pt")
    torch.save(empty_idx.cpu(),               output_dir / "val_idx.pt")
    torch.save(empty_idx.cpu(),               output_dir / "test_idx.pt")
    torch.save(edge_index.cpu(),              output_dir / "edge_index.pt")
    torch.save(edge_type.cpu(),               output_dir / "edge_type.pt")
    torch.save(num_properties_tensor.cpu(),   output_dir / "num_properties_tensor.pt")
    torch.save(cat_properties_tensor.cpu(),   output_dir / "cat_properties_tensor.pt")
    torch.save(des_tensor.cpu(),              output_dir / "des_tensor.pt")
    torch.save(tweets_tensor.cpu(),           output_dir / "tweets_tensor.pt")

    save_json(output_dir / "all_uids.json", all_uids)

    # ── [6/6] 保存 metadata ───────────────────────────────────────────────────
    posts_count = sum(len(v) for v in posts_by_author.values())
    metadata = {
        "dataset": "Moltbook",
        "purpose": "OOD evaluation — all users are agents (bots), ground truth = 1",
        "num_total_nodes": N,
        "num_labeled_users": 0,
        "num_agent_users": N,
        "num_train": 0,
        "num_val": 0,
        "num_test": 0,
        "edge_count": int(edge_index.shape[1]),
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
            "interactions_count_proxy_karma",
        ],
        "num_feature_mapping": {
            "followers_count": "user_info.followers",
            "active_days": "join_time ('%Y-%m-%d') to reference_date",
            "screen_name_length": "len(author)",
            "friends_count": "user_info.following",
            "interactions_count_proxy": "user_info.karma (confirmed by user)",
        },
        "cat_feature_order": ["default_profile_image"],
        "cat_feature_mapping": {
            "default_profile_image": "all ones (Moltbook accounts all use default avatar)",
        },
        "graph_construction": {
            "method": "reply_graph",
            "source": "comment_records.jsonl + post_records.jsonl",
            "rule": "directed edge (commenter -> post_author); deduplicated; no self-loops; both nodes must be in user_info",
        },
        "normalization": {
            "method": "z-score using Weibo training set statistics",
            "stats_source": str(weibo_proc_dir / "metadata.json"),
            "num_feature_mean": mean.tolist(),
            "num_feature_std": std.tolist(),
        },
        "text_encoding": {
            "model": str(MODEL_DIR),
            "description_field": "discription (note: Moltbook typo)",
            "post_field": "user_history_posts.jsonl short_content (mean pooling)",
        },
        "label_info": {
            "all_labels": 1,
            "meaning": "all agents are bots",
        },
        "posts_with_text_count": posts_count,
    }
    save_json(output_dir / "metadata.json", metadata)

    print("\n[6/6] Done.")
    print(f"  label              : {tuple(labels_tensor.shape)}")
    print(f"  edge_index         : {tuple(edge_index.shape)}")
    print(f"  num_properties     : {tuple(num_properties_tensor.shape)}")
    print(f"  cat_properties     : {tuple(cat_properties_tensor.shape)}")
    print(f"  des_tensor         : {tuple(des_tensor.shape)}")
    print(f"  tweets_tensor      : {tuple(tweets_tensor.shape)}")
    print(f"  Output dir         : {output_dir}")


if __name__ == "__main__":
    main()
