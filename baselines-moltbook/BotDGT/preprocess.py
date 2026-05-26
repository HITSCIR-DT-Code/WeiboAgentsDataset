"""
  cd baselines-moltbook/BotDGT
  python preprocess.py
  python preprocess.py --weibo-raw-dir ../../dataset/Weibo \\
      --weibo-graph-dir ../../baselines-weibo/BotDGT/data/Weibo/graph_data/graphs \\
      --moltbook-dir ../../dataset/Moltbook \\
      --output-dir ./data/Moltbook \\
      --device auto --batch-size 32 --max-posts 20
"""

import argparse
import json
import os
import pickle
import datetime
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import tqdm
from transformers import AutoTokenizer, AutoModel

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
ROOT_DIR       = BASE_DIR.parent.parent
MODEL_DIR      = ROOT_DIR / "models" / "xlm-roberta-base"

# 数据采集截止日期（与 Weibo preprocess.py 保持一致，计算账户年限时使用）
DATA_CUTOFF = datetime.date(2026, 4, 1)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Moltbook data into BotDGT format for OOD evaluation."
    )
    parser.add_argument(
        "--weibo-raw-dir", type=str,
        default=str(ROOT_DIR / "dataset" / "Weibo"),
        help="Weibo 原始数据目录，用于计算归一化统计量（默认：dataset/Weibo）",
    )
    parser.add_argument(
        "--weibo-graph-dir", type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "BotDGT" / "data" / "Weibo" / "graph_data" / "graphs"),
        help="Weibo 时序快照目录，用于对齐 Moltbook 快照时间轴（默认：baselines-weibo/BotDGT/data/Weibo/graph_data/graphs）",
    )
    parser.add_argument(
        "--moltbook-dir", type=str,
        default=str(ROOT_DIR / "dataset" / "Moltbook"),
        help="Moltbook 原始数据目录（默认：dataset/Moltbook）",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(BASE_DIR / "data" / "Moltbook"),
        help="预处理结果输出目录（默认：data/Moltbook）",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="BERT 编码设备：auto / cuda / mps / cpu（默认：auto）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="BERT 编码批大小（默认：32）",
    )
    parser.add_argument(
        "--max-posts", type=int, default=20,
        help="每个用户最多使用的历史帖子数量（默认：20）",
    )
    return parser.parse_args()


# ── 设备检测 ──────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"当前环境不支持 CUDA，但收到 device={requested}")
        return requested
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(f"当前环境不支持 MPS，但收到 device={requested}")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"不支持的 device: {requested}")


# ── I/O 工具 ──────────────────────────────────────────────────────────────────

def iterate_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ============================================================
# Step 1: 加载 Moltbook 用户信息
# ============================================================

def load_moltbook_users(moltbook_dir: str):
    """读取 user_info.jsonl，返回用户列表和 author→index 映射。"""
    users = []
    for record in iterate_jsonl(os.path.join(moltbook_dir, "user_info.jsonl")):
        users.append(record)
    uid2index = {user["author"]: idx for idx, user in enumerate(users)}
    print(f"Moltbook 用户数: {len(users)}")
    return users, uid2index


# ============================================================
# Step 2: 从原始 Weibo 数据计算归一化统计量
# ============================================================

def compute_weibo_norm_stats(weibo_raw_dir: str):
    """
    从原始 Weibo 用户数据（排除 label=1 的疑似用户）重新计算 5 维数值特征的
    mean 和 std，用于 z-score 归一化。与 Weibo BotDGT preprocess.py 中等效。

    特征顺序（与 BotDGT num_prop 对应）：
      [0] followers_count
      [1] active_days
      [2] screen_name_length
      [3] friends_count
      [4] interactions_count
    """
    labels_path = os.path.join(weibo_raw_dir, "Weibo_Labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)

    excluded_uids = {uid for uid, lb in raw_labels.items() if lb == 1}
    print(f"Weibo 排除 label=1 用户: {len(excluded_uids)} 个")

    rows = []
    for record in iterate_jsonl(os.path.join(weibo_raw_dir, "user_profile.jsonl")):
        if record["uid"] in excluded_uids:
            continue
        try:
            created = datetime.datetime.strptime(
                record["created_at"], "%Y-%m-%d %H:%M:%S").date()
        except (ValueError, TypeError):
            created = datetime.date(2000, 1, 1)
        active_days = max(0, (DATA_CUTOFF - created).days)
        rows.append([
            float(record.get("followers_count", 0)),
            float(active_days),
            float(len(record.get("screen_name", "") or "")),
            float(record.get("friends_count", 0)),
            float(record.get("interactions_count", 0)),
        ])

    arr = np.array(rows, dtype=np.float32)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0

    print(f"Weibo 归一化统计量（基于 {len(rows)} 用户）:")
    feature_names = ["followers_count", "active_days", "screen_name_length",
                     "friends_count", "interactions_count"]
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={mean[i]:.4f}, std={std[i]:.4f}")

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


# ============================================================
# Step 3: 数值特征 num_prop [N, 5]
# ============================================================

def compute_num_prop(users, norm_mean: torch.Tensor, norm_std: torch.Tensor):
    """
    计算 Moltbook 用户 5 维数值特征，并用 Weibo 的 mean/std 做 z-score 归一化。
    特征顺序与 Weibo BotDGT 训练时完全一致。
    """
    rows = []
    for user in users:
        try:
            created = datetime.datetime.strptime(
                user["join_time"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            created = datetime.date(2000, 1, 1)
        active_days = max(0, (DATA_CUTOFF - created).days)
        rows.append([
            float(user.get("followers", 0)),
            float(active_days),
            float(len(user.get("author", "") or "")),
            float(user.get("following", 0)),
            float(user.get("karma", 0)),
        ])
    arr = np.array(rows, dtype=np.float32)
    mean_np = norm_mean.numpy()
    std_np  = norm_std.numpy()
    arr = (arr - mean_np) / std_np
    return torch.tensor(arr, dtype=torch.float32)


# ============================================================
# Step 4: 类别特征 cat_prop [N, 1]
# ============================================================

def compute_cat_prop(users):
    """类别特征：verified 布尔值（与 Weibo BotDGT 训练时一致）。"""
    rows = [[float(bool(user.get("verified", False)))] for user in users]
    return torch.tensor(rows, dtype=torch.float32)


# ============================================================
# Step 5: 文本嵌入（xlm-roberta-base）
# ============================================================

def encode_texts_bert(texts, tokenizer, model, device, batch_size=32):
    """对文本列表进行 BERT mean-pooling 编码，返回 [N, 768]。"""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="BERT encoding"):
            batch = texts[i: i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            attention_mask = encoded["attention_mask"]           # [B, L]
            token_emb = output.last_hidden_state                 # [B, L, 768]
            mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
            sum_emb  = torch.sum(token_emb * mask_exp, dim=1)
            sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            embeddings = sum_emb / sum_mask
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def compute_des_tensor(users, tokenizer, model, device, batch_size=32):
    """description 文本嵌入 [N, 768]；注意字段名为 discription（拼写错误）。"""
    texts = [user.get("discription") or "" for user in users]
    return encode_texts_bert(texts, tokenizer, model, device, batch_size)


def compute_tweets_tensor(users, uid2index, moltbook_dir, tokenizer, model,
                          device, batch_size=32, max_posts=20):
    """
    从 user_history_posts.jsonl 读取历史帖子，每用户最多取 max_posts 条，
    拼接 title + " " + short_content 后编码为 [N, 768]。
    无帖子的用户用零向量。
    """
    uid_texts = {user["author"]: [] for user in users}

    print("读取历史帖子数据（user_history_posts.jsonl）...")
    hist_path = os.path.join(moltbook_dir, "user_history_posts.jsonl")
    for record in tqdm.tqdm(iterate_jsonl(hist_path), desc="Loading posts"):
        author = record.get("author")
        if author in uid_texts and len(uid_texts[author]) < max_posts:
            title = (record.get("title") or "").strip()
            short_content = (record.get("short_content") or "").strip()
            text = (title + " " + short_content).strip()
            if text:
                uid_texts[author].append(text)

    N = len(users)
    valid_pairs = []
    for idx, user in enumerate(users):
        texts = uid_texts[user["author"]]
        if texts:
            combined = " ".join(texts)[:512]
            valid_pairs.append((idx, combined))

    no_post_count = N - len(valid_pairs)
    print(f"有历史帖子用户: {len(valid_pairs)}, 无历史帖子用户: {no_post_count}")

    result = torch.zeros(N, 768, dtype=torch.float32)
    if valid_pairs:
        indices, text_list = zip(*valid_pairs)
        embeddings = encode_texts_bert(list(text_list), tokenizer, model, device, batch_size)
        for i, idx in enumerate(indices):
            result[idx] = embeddings[i]
    return result


# ============================================================
# Step 6: 回复图 edge_index [2, E], edge_type [E]
# ============================================================

def compute_edges(users, uid2index, moltbook_dir):
    """
    从 comment_records.jsonl 和 post_records.jsonl 构建回复关系图。
    边方向：commenter → post_author（即回复者 → 被回复帖子的作者）。
    去重：只有无有，不考虑回复次数。
    """
    # 建立 post_url → author 映射
    url2author = {}
    for record in iterate_jsonl(os.path.join(moltbook_dir, "post_records.jsonl")):
        url = record.get("url")
        author = record.get("author")
        if url and author:
            url2author[url] = author

    # 遍历评论，构建边集合（去重）
    edge_set = set()
    for record in iterate_jsonl(os.path.join(moltbook_dir, "comment_records.jsonl")):
        commenter = record.get("author")
        post_url  = record.get("post_url")
        if not commenter or not post_url:
            continue
        post_author = url2author.get(post_url)
        if post_author is None:
            continue
        if commenter not in uid2index or post_author not in uid2index:
            continue
        if commenter == post_author:
            continue
        src = uid2index[commenter]
        dst = uid2index[post_author]
        edge_set.add((src, dst))

    if edge_set:
        src_list, dst_list = zip(*edge_set)
        src_list = list(src_list)
        dst_list = list(dst_list)
    else:
        src_list, dst_list = [], []

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type  = torch.zeros(len(src_list), dtype=torch.long)
    print(f"回复图边数（去重后）: {len(src_list)}")
    return edge_index, edge_type


# ============================================================
# Step 7: 标签 & 数据集划分
# ============================================================

def compute_labels_and_splits(users):
    """
    label.pt 大小 = N，值全为 3（OOD 占位符，与 Weibo 中 Agent 用户一致）。
    ood_idx.pt 包含所有用户索引。
    train/val/test idx 均为空 tensor。
    """
    N = len(users)
    label_tensor = torch.full((N,), 3, dtype=torch.long)
    ood_idx      = torch.arange(N, dtype=torch.long)
    train_idx    = torch.tensor([], dtype=torch.long)
    val_idx      = torch.tensor([], dtype=torch.long)
    test_idx     = torch.tensor([], dtype=torch.long)
    print(f"标签: 全 3（OOD）, 共 {N} 个用户")
    return label_tensor, train_idx, val_idx, test_idx, ood_idx


# ============================================================
# Step 8: 月粒度时序图快照（与 Weibo 快照时间轴完全对齐）
# ============================================================

def generate_graph_snapshots(users, uid2index, edge_index, edge_type,
                             weibo_graph_dir: str, output_graph_dir: str):
    """
    从 weibo_graph_dir 读取快照文件名列表，以获取与 Weibo 完全一致的时间轴。
    对每个快照日期，生成对应的 Moltbook 快照文件。

    用户的加入时间（join_time，格式 "%Y-%m-%d"）用于确定用户是否在该快照存在。
    快照内只保留 join_time < snapshot_date 的用户（与 Weibo 中 created_at < snapshot_date 一致）。
    """
    N = len(users)

    # 从文件名解析快照日期列表（格式 graph_in_snapshot_YYYY-MM-DD.pt）
    weibo_files = sorted(os.listdir(weibo_graph_dir))
    weibo_files = [f for f in weibo_files if f.endswith(".pt") and f.startswith("graph_in_snapshot_")]
    if not weibo_files:
        raise FileNotFoundError(f"在 {weibo_graph_dir} 中未找到快照文件，请先运行 Weibo BotDGT preprocess.py")

    print(f"\n与 Weibo 对齐：共 {len(weibo_files)} 个快照")
    print(f"  {weibo_files[0]} → {weibo_files[-1]}")

    # 解析每个用户的加入日期（ordinal，方便比较）
    user_join_dates = []
    for user in users:
        try:
            d = datetime.datetime.strptime(user["join_time"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            d = datetime.date(2000, 1, 1)
        user_join_dates.append(d.toordinal())
    user_join_dates = np.array(user_join_dates, dtype=np.int64)

    # 预处理边数组（numpy，便于快速过滤）
    if edge_index.shape[1] > 0:
        ei_np = edge_index.numpy()  # [2, E]
    else:
        ei_np = np.zeros((2, 0), dtype=np.int64)

    from torch_geometric.data import Data

    for fname in tqdm.tqdm(weibo_files, desc="Generating Moltbook snapshots"):
        # 解析快照日期：graph_in_snapshot_YYYY-MM-DD.pt → YYYY-MM-DD
        date_str = fname[len("graph_in_snapshot_"):-len(".pt")]
        try:
            snapshot_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"  跳过无法解析日期的文件: {fname}")
            continue

        snap_ord = snapshot_date.toordinal()

        # 存在于该快照的节点：join_time < snapshot_date
        exist_mask = user_join_dates < snap_ord  # [N] bool

        # 过滤边（双方都必须 exist）
        if ei_np.shape[1] > 0 and exist_mask.sum() > 0:
            valid_edge = exist_mask[ei_np[0]] & exist_mask[ei_np[1]]
            snap_src = ei_np[0][valid_edge]
            snap_dst = ei_np[1][valid_edge]
        else:
            snap_src = np.array([], dtype=np.int64)
            snap_dst = np.array([], dtype=np.int64)

        snap_edge_index = (
            torch.tensor(np.stack([snap_src, snap_dst]), dtype=torch.long)
            if len(snap_src) > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )
        snap_edge_type = torch.zeros(len(snap_src), dtype=torch.long)

        # exist_nodes [N, 1] float
        exist_nodes = torch.tensor(exist_mask, dtype=torch.float32)

        # 位置编码：clustering_coefficient 和 bidirectional_links_ratio [N, 1]
        clustering_coeff    = torch.zeros(N, 1, dtype=torch.float32)
        bidirectional_ratio = torch.zeros(N, 1, dtype=torch.float32)

        if len(snap_src) > 0:
            exist_indices = np.where(exist_mask)[0].tolist()

            G_ud = nx.Graph()
            G_ud.add_nodes_from(exist_indices)
            G_ud.add_edges_from(zip(snap_src.tolist(), snap_dst.tolist()))

            G_d = nx.DiGraph()
            G_d.add_nodes_from(exist_indices)
            G_d.add_edges_from(zip(snap_src.tolist(), snap_dst.tolist()))

            cc = nx.clustering(G_ud)
            for node, val in cc.items():
                clustering_coeff[node, 0] = float(val)

            for node in exist_indices:
                out_neighbors = set(G_d.successors(node))
                in_neighbors  = set(G_d.predecessors(node))
                out_deg = len(out_neighbors)
                if out_deg > 0:
                    bi = len(out_neighbors & in_neighbors) / out_deg
                    bidirectional_ratio[node, 0] = float(bi)

        data = Data(
            edge_index=snap_edge_index,
            edge_type=snap_edge_type,
            exist_nodes=exist_nodes,
            clustering_coefficient=clustering_coeff,
            bidirectional_links_ratio=bidirectional_ratio,
            n_id=torch.arange(N, dtype=torch.long),
        )

        save_path = os.path.join(output_graph_dir, fname)
        torch.save(data, save_path)

    print(f"快照已保存至 {output_graph_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"使用设备: {device}")

    output_dir       = Path(args.output_dir)
    processed_dir    = output_dir / "processed_data"
    output_graph_dir = output_dir / "graph_data" / "graphs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_graph_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: 加载 Moltbook 用户信息")
    users, uid2index = load_moltbook_users(args.moltbook_dir)

    with open(processed_dir / "uid2global_index.pkl", "wb") as f:
        pickle.dump(uid2index, f)
    print(f"uid2global_index 已保存 ({len(uid2index)} 用户)")

    print("\nStep 2: 从 Weibo 原始数据计算归一化统计量")
    norm_mean, norm_std = compute_weibo_norm_stats(args.weibo_raw_dir)
    torch.save({"mean": norm_mean, "std": norm_std}, processed_dir / "norm_stats.pt")
    print("norm_stats 已保存")

    print("\nStep 3: 数值特征 num_prop [N, 5]")
    num_prop = compute_num_prop(users, norm_mean, norm_std)
    torch.save(num_prop, processed_dir / "num_properties_tensor.pt")
    print(f"num_prop shape: {num_prop.shape}")

    print("\nStep 4: 类别特征 cat_prop [N, 1]")
    cat_prop = compute_cat_prop(users)
    torch.save(cat_prop, processed_dir / "cat_properties_tensor.pt")
    print(f"cat_prop shape: {cat_prop.shape}")

    print("\nStep 5: 文本嵌入（xlm-roberta-base）")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    bert_model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device)

    print("  计算 des_tensor（description 嵌入）...")
    des_tensor = compute_des_tensor(users, tokenizer, bert_model, device, args.batch_size)
    torch.save(des_tensor, processed_dir / "des_tensor.pt")
    print(f"  des_tensor shape: {des_tensor.shape}")

    print("  计算 tweets_tensor（历史帖子嵌入）...")
    tweets_tensor = compute_tweets_tensor(
        users, uid2index, args.moltbook_dir,
        tokenizer, bert_model, device, args.batch_size, args.max_posts,
    )
    torch.save(tweets_tensor, processed_dir / "tweets_tensor.pt")
    print(f"  tweets_tensor shape: {tweets_tensor.shape}")

    del bert_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nStep 6: 构建回复图边")
    edge_index, edge_type = compute_edges(users, uid2index, args.moltbook_dir)
    torch.save(edge_index, processed_dir / "edge_index.pt")
    torch.save(edge_type,  processed_dir / "edge_type.pt")
    print(f"edge_index shape: {edge_index.shape}")

    print("\nStep 7: 生成标签和数据集划分")
    label_tensor, train_idx, val_idx, test_idx, ood_idx = compute_labels_and_splits(users)
    torch.save(label_tensor, processed_dir / "label.pt")
    torch.save(train_idx,    processed_dir / "train_idx.pt")
    torch.save(val_idx,      processed_dir / "val_idx.pt")
    torch.save(test_idx,     processed_dir / "test_idx.pt")
    torch.save(ood_idx,      processed_dir / "ood_idx.pt")
    print("标签和划分已保存")

    print("\nStep 8: 生成月粒度时序图快照（与 Weibo 对齐）")
    generate_graph_snapshots(
        users, uid2index, edge_index, edge_type,
        args.weibo_graph_dir, str(output_graph_dir),
    )

    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"  特征文件目录: {processed_dir}")
    print(f"  图快照目录:   {output_graph_dir}")


if __name__ == "__main__":
    main()
