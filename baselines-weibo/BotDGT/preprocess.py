"""
    python preprocess.py
"""

import json
import os
import pickle
import datetime
from collections import Counter

import numpy as np
import torch
import networkx as nx
import tqdm
from transformers import AutoTokenizer, AutoModel


def resolve_device(device='auto'):
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError(f'当前环境不支持 CUDA，但收到 device={device}')
        return device

    if device.startswith('mps'):
        if not torch.backends.mps.is_available():
            raise ValueError(f'当前环境不支持 MPS，但收到 device={device}')
        return 'mps'

    if device == 'cpu':
        return device

    raise ValueError(f'不支持的 device: {device}')

# ---- 路径配置 ----
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
WEIBO_DIR     = os.path.join(BASE_DIR, '../../dataset/Weibo')
BERT_DIR      = os.path.join(BASE_DIR, '../../models/xlm-roberta-base')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data/Weibo/processed_data')
GRAPH_DIR     = os.path.join(BASE_DIR, 'data/Weibo/graph_data/graphs')

# 数据采集截止日期（用于计算账户活跃天数）
DATA_CUTOFF = datetime.date(2026, 4, 1)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# Step 1: 加载用户信息和标签，过滤 label=1 的疑似用户
# ============================================================

def load_users_and_labels():
    """返回 users(list), uid2train_label(dict: uid -> 训练标签 0/1/3(OOD))"""
    with open(os.path.join(WEIBO_DIR, 'Weibo_Labels.json'), 'r', encoding='utf-8') as f:
        raw_labels = json.load(f)

    print("Weibo_Labels.json 标签分布:")
    cnt = Counter(raw_labels.values())
    for k in sorted(cnt):
        meaning = {0: '人类', 1: '疑似(排除)', 2: '机器人', -1: 'Agent(OOD)'}
        print(f"  {k:3d} ({meaning.get(k, '?')}): {cnt[k]}")

    excluded_uids = {uid for uid, lb in raw_labels.items() if lb == 1}
    print(f"\n排除 label=1 的疑似用户: {len(excluded_uids)} 个")

    users = []
    with open(os.path.join(WEIBO_DIR, 'user_profile.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user = json.loads(line)
            if user['uid'] in excluded_uids:
                continue
            users.append(user)
    print(f"保留用户数: {len(users)}")

    uid2train_label = {}
    for uid, lb in raw_labels.items():
        if uid in excluded_uids:
            continue
        if lb == 0:
            uid2train_label[uid] = 0   # 人类
        elif lb == 2:
            uid2train_label[uid] = 1   # 机器人
        elif lb == -1:
            uid2train_label[uid] = 3   # OOD (Agent)
        # label=1 用户已排除，不会出现在这里

    return users, uid2train_label


def build_uid2index(users):
    return {user['uid']: idx for idx, user in enumerate(users)}


def load_official_split():
    """返回官方划分字典：uid -> train/val/test/ood"""
    split_path = os.path.join(WEIBO_DIR, 'split.json')
    with open(split_path, 'r', encoding='utf-8') as f:
        split_map = json.load(f)

    valid_splits = {'train', 'val', 'test', 'ood'}
    invalid_items = {uid: split for uid, split in split_map.items() if split not in valid_splits}
    if invalid_items:
        sample = list(invalid_items.items())[:5]
        raise ValueError(f'split.json 中存在非法划分标签: {sample}')

    split_counter = Counter(split_map.values())
    print('官方 split.json 划分统计:')
    for split_name in ['train', 'val', 'test', 'ood']:
        print(f'  {split_name:5s}: {split_counter.get(split_name, 0)}')

    return split_map


# ============================================================
# Step 2: 数值特征 num_prop [N, 5]
# 字段: followers_count, active_days, len(screen_name),
#        friends_count, interactions_count
# ============================================================

def compute_num_prop(users):
    rows = []
    for user in users:
        try:
            created = datetime.datetime.strptime(
                user['created_at'], '%Y-%m-%d %H:%M:%S').date()
        except (ValueError, TypeError):
            created = datetime.date(2000, 1, 1)
        active_days = max(0, (DATA_CUTOFF - created).days)
        rows.append([
            float(user.get('followers_count', 0)),
            float(active_days),
            float(len(user.get('screen_name', '') or '')),
            float(user.get('friends_count', 0)),
            float(user.get('interactions_count', 0)),
        ])
    arr = np.array(rows, dtype=np.float32)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    arr = (arr - mean) / std
    return torch.tensor(arr, dtype=torch.float32)


# ============================================================
# Step 3: 类别特征 cat_prop [N, 1]
# 字段: verified（仅保留有数据的字段，不占位）
# ============================================================

def compute_cat_prop(users):
    rows = [[float(bool(user.get('verified', False)))] for user in users]
    return torch.tensor(rows, dtype=torch.float32)


# ============================================================
# Step 4: 文本嵌入（用 xlm-roberta-base）
# ============================================================

def encode_texts_bert(texts, tokenizer, model, device, batch_size=64):
    """对文本列表进行 BERT mean-pooling 编码，返回 [N, 768]"""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc='BERT encoding'):
            batch = texts[i: i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            ).to(device)
            output = model(**encoded)
            attention_mask = encoded['attention_mask']          # [B, L]
            token_emb = output.last_hidden_state                # [B, L, 768]
            mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
            sum_emb  = torch.sum(token_emb * mask_exp, dim=1)
            sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            embeddings = sum_emb / sum_mask
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def compute_des_tensor(users, tokenizer, model, device):
    texts = [user.get('description') or '' for user in users]
    return encode_texts_bert(texts, tokenizer, model, device)


def compute_tweets_tensor(users, tokenizer, model, device):
    """
    读取 user_tweets.jsonl，每用户所有推文（原创+转发）拼接后编码
    无推文的用户用零向量
    """
    uid_set = {user['uid'] for user in users}
    uid_texts = {user['uid']: [] for user in users}

    print("读取推文数据...")
    with open(os.path.join(WEIBO_DIR, 'user_tweets.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tweet = json.loads(line)
            uid = tweet.get('uid')
            if uid in uid_texts:
                text = (tweet.get('text') or '').strip()
                if text:
                    uid_texts[uid].append(text)

    N = len(users)
    # 收集需要编码的文本（含用户下标）
    valid_pairs = []
    for idx, user in enumerate(users):
        user_texts = uid_texts[user['uid']]
        if user_texts:
            combined = ' '.join(user_texts)[:512]
            valid_pairs.append((idx, combined))

    no_tweet_count = N - len(valid_pairs)
    print(f"有推文用户: {len(valid_pairs)}, 无推文用户: {no_tweet_count}")

    result = torch.zeros(N, 768, dtype=torch.float32)
    if valid_pairs:
        indices, text_list = zip(*valid_pairs)
        embeddings = encode_texts_bert(list(text_list), tokenizer, model, device)
        for i, idx in enumerate(indices):
            result[idx] = embeddings[i]
    return result


# ============================================================
# Step 5: 关注图  edge_index [2, E], edge_type [E]
# ============================================================

def compute_edges(uid2index):
    src_list, dst_list = [], []
    with open(os.path.join(WEIBO_DIR, 'followings.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edge = json.loads(line)
            src = edge.get('source_uid')
            dst = edge.get('target_uid')
            if src in uid2index and dst in uid2index:
                src_list.append(uid2index[src])
                dst_list.append(uid2index[dst])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type  = torch.zeros(len(src_list), dtype=torch.long)  # 类型 0 = following
    print(f"边数量（过滤后）: {len(src_list)}")
    return edge_index, edge_type


# ============================================================
# Step 6: 标签张量 + 数据集划分
# ============================================================

def compute_labels_and_splits(users, uid2train_label, official_split):
    """
    label.pt 大小 = N, 值域: 0(人类), 1(机器人), 3(OOD/无标签)
    train/val/test 仅来自 label=0 和 label=1 的用户，且严格使用官方 split.json
    OOD 索引来自 uid2train_label 中值为 3 的用户
    """
    N = len(users)
    label_tensor = torch.full((N,), 3, dtype=torch.long)
    for idx, user in enumerate(users):
        uid = user['uid']
        if uid in uid2train_label:
            label_tensor[idx] = uid2train_label[uid]

    split_indices = {'train': [], 'val': [], 'test': []}
    ood_idx = []
    missing_split_uids = []
    invalid_split_assignments = []

    for idx, user in enumerate(users):
        uid = user['uid']
        train_label = uid2train_label.get(uid)

        if train_label in (0, 1):
            split_name = official_split.get(uid)
            if split_name is None:
                missing_split_uids.append(uid)
                continue
            split_indices[split_name].append(idx)
        elif train_label == 3:
            ood_idx.append(idx)

    for uid, split_name in official_split.items():
        train_label = uid2train_label.get(uid)
        if split_name == 'ood':
            if train_label != 3:
                invalid_split_assignments.append((uid, split_name, train_label))
        elif train_label not in (0, 1):
            invalid_split_assignments.append((uid, split_name, train_label))

    if missing_split_uids:
        sample = missing_split_uids[:10]
        raise ValueError(
            '存在可训练用户未出现在官方 split.json 中，无法构造固定划分。'
            f'示例 uid: {sample}'
        )

    if invalid_split_assignments:
        sample = invalid_split_assignments[:10]
        raise ValueError(
            '官方 split.json 中存在非法的标签-划分组合。'
            'train/val/test 只能包含人工标注的 0/2 类样本，ood 只能包含 Agent 用户。'
            f'示例: {sample}'
        )

    tr_idx = split_indices['train']
    va_idx = split_indices['val']
    te_idx = split_indices['test']

    print(f"\n数据集划分:")
    print(f"  Train: {len(tr_idx)}  (标签分布: {Counter(label_tensor[tr_idx].tolist())})")
    print(f"  Val:   {len(va_idx)}  (标签分布: {Counter(label_tensor[va_idx].tolist())})")
    print(f"  Test:  {len(te_idx)}  (标签分布: {Counter(label_tensor[te_idx].tolist())})")
    print(f"  OOD:   {len(ood_idx)}")

    return (
        label_tensor,
        torch.tensor(tr_idx,  dtype=torch.long),
        torch.tensor(va_idx,  dtype=torch.long),
        torch.tensor(te_idx,  dtype=torch.long),
        torch.tensor(ood_idx, dtype=torch.long),
    )


# ============================================================
# Step 7: 月粒度时序图快照
# ============================================================

def generate_graph_snapshots(users, uid2index, edge_index, edge_type):
    N = len(users)

    # 解析注册日期
    uid_dates = {}
    for user in users:
        try:
            d = datetime.datetime.strptime(
                user['created_at'], '%Y-%m-%d %H:%M:%S').date()
        except (ValueError, TypeError):
            d = datetime.date(2000, 1, 1)
        uid_dates[user['uid']] = d

    min_date = min(uid_dates.values())

    # 生成月份序列：从最早注册月到 DATA_CUTOFF
    months = []
    y, m = min_date.year, min_date.month
    ey, em = DATA_CUTOFF.year, DATA_CUTOFF.month
    while (y, m) <= (ey, em):
        months.append(datetime.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1

    print(f"\n时序快照: {len(months)} 个月，{months[0]} → {months[-1]}")

    # 预转换 edge_index 为 numpy（便于快速过滤）
    if edge_index.shape[1] > 0:
        ei_np = edge_index.numpy()  # [2, E]
    else:
        ei_np = np.zeros((2, 0), dtype=np.int64)

    # 预计算每个用户的注册日期列表（按 index 排列）
    user_dates_arr = np.array(
        [uid_dates[user['uid']].toordinal() for user in users], dtype=np.int64
    )

    for snapshot_date in tqdm.tqdm(months, desc='Generating snapshots'):
        snap_ord = snapshot_date.toordinal()

        # 存在于该快照的节点：created_at < snapshot_date
        exist_mask = user_dates_arr < snap_ord   # [N] bool
        if exist_mask.sum() == 0:
            continue

        # 过滤边
        if ei_np.shape[1] > 0:
            src_arr = ei_np[0]
            dst_arr = ei_np[1]
            valid_edge = exist_mask[src_arr] & exist_mask[dst_arr]
            snap_src = src_arr[valid_edge]
            snap_dst = dst_arr[valid_edge]
        else:
            snap_src = np.array([], dtype=np.int64)
            snap_dst = np.array([], dtype=np.int64)

        snap_edge_index = torch.tensor(
            np.stack([snap_src, snap_dst]), dtype=torch.long
        ) if len(snap_src) > 0 else torch.zeros((2, 0), dtype=torch.long)
        snap_edge_type = torch.zeros(len(snap_src), dtype=torch.long)

        # exist_nodes [N, 1] float — 注意：PositionEmbeddingLayer 需要 [N, 1]
        exist_nodes = torch.tensor(exist_mask, dtype=torch.float32)

        # 位置编码：聚类系数 & 双向链接比率 [N, 1]
        clustering_coeff      = torch.zeros(N, 1, dtype=torch.float32)
        bidirectional_ratio   = torch.zeros(N, 1, dtype=torch.float32)

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

        from torch_geometric.data import Data
        data = Data(
            edge_index=snap_edge_index,
            edge_type=snap_edge_type,
            exist_nodes=exist_nodes,
            clustering_coefficient=clustering_coeff,
            bidirectional_links_ratio=bidirectional_ratio,
            n_id=torch.arange(N, dtype=torch.long),
        )

        save_path = os.path.join(GRAPH_DIR, f'graph_in_snapshot_{snapshot_date}.pt')
        torch.save(data, save_path)

    print(f"快照已保存至 {GRAPH_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Step 1: 加载用户和标签")
    users, uid2train_label = load_users_and_labels()
    official_split = load_official_split()
    uid2index = build_uid2index(users)

    with open(os.path.join(PROCESSED_DIR, 'uid2global_index.pkl'), 'wb') as f:
        pickle.dump(uid2index, f)
    print(f"uid2global_index 已保存 ({len(uid2index)} 用户)")

    print("\nStep 2: 数值特征 num_prop")
    num_prop = compute_num_prop(users)
    torch.save(num_prop, os.path.join(PROCESSED_DIR, 'num_properties_tensor.pt'))
    print(f"num_prop shape: {num_prop.shape}")

    print("\nStep 3: 类别特征 cat_prop")
    cat_prop = compute_cat_prop(users)
    torch.save(cat_prop, os.path.join(PROCESSED_DIR, 'cat_properties_tensor.pt'))
    print(f"cat_prop shape: {cat_prop.shape}")

    print("\nStep 4: 文本嵌入（xlm-roberta-base）")
    device = resolve_device()
    print(f"使用设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_model = AutoModel.from_pretrained(BERT_DIR).to(device)

    print("  计算 des_tensor...")
    des_tensor = compute_des_tensor(users, tokenizer, bert_model, device)
    torch.save(des_tensor, os.path.join(PROCESSED_DIR, 'des_tensor.pt'))
    print(f"  des_tensor shape: {des_tensor.shape}")

    print("  计算 tweets_tensor...")
    tweets_tensor = compute_tweets_tensor(users, tokenizer, bert_model, device)
    torch.save(tweets_tensor, os.path.join(PROCESSED_DIR, 'tweets_tensor.pt'))
    print(f"  tweets_tensor shape: {tweets_tensor.shape}")

    del bert_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("\nStep 5: 构建关注图边")
    edge_index, edge_type = compute_edges(uid2index)
    torch.save(edge_index, os.path.join(PROCESSED_DIR, 'edge_index.pt'))
    torch.save(edge_type,  os.path.join(PROCESSED_DIR, 'edge_type.pt'))
    print(f"edge_index shape: {edge_index.shape}")

    print("\nStep 6: 生成标签和数据集划分")
    label_tensor, train_idx, val_idx, test_idx, ood_idx = \
        compute_labels_and_splits(users, uid2train_label, official_split)
    torch.save(label_tensor, os.path.join(PROCESSED_DIR, 'label.pt'))
    torch.save(train_idx,    os.path.join(PROCESSED_DIR, 'train_idx.pt'))
    torch.save(val_idx,      os.path.join(PROCESSED_DIR, 'val_idx.pt'))
    torch.save(test_idx,     os.path.join(PROCESSED_DIR, 'test_idx.pt'))
    torch.save(ood_idx,      os.path.join(PROCESSED_DIR, 'ood_idx.pt'))

    print("\nStep 7: 生成月粒度时序图快照")
    generate_graph_snapshots(users, uid2index, edge_index, edge_type)

    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"处理后的特征文件: {PROCESSED_DIR}")
    print(f"图快照文件:       {GRAPH_DIR}")


if __name__ == '__main__':
    main()
