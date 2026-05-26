"""
python preprocess.py
"""

import os
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import entropy, bigrams_freq

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "dataset" / "Weibo"
BERT_PATH = BASE_DIR.parent.parent / "models" / "xlm-roberta-base"
SAVE_DIR = BASE_DIR / "processed"
SAVE_DIR.mkdir(exist_ok=True)

# 基准日期：数据集截止日期（近似）
DATE_REF = datetime(2025, 4, 1)

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_labels():
    with open(DATA_DIR / "Weibo_Labels.json", "r", encoding="utf-8") as f:
        label_dict = json.load(f)  # {uid: label}
    return {str(k): int(v) for k, v in label_dict.items()}


def load_profiles():
    profiles = {}
    with open(DATA_DIR / "user_profile.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj.get("uid", ""))
            if uid:
                profiles[uid] = obj
    return profiles

# ── 特征提取 ──────────────────────────────────────────────────────────────────

def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def compute_active_days(created_at_str):
    """从 created_at 字符串计算截止 DATE_REF 的天数"""
    if not created_at_str:
        return 1
    try:
        dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
        days = (DATE_REF - dt).days
        return max(days, 1)
    except ValueError:
        return 1


def extract_num_features(uids, profiles):
    """
    提取 9 个数值特征，返回 numpy array [N, 9]

    特征顺序（与原版对齐）：
      0: followers_count
      1: interactions_count / active_days  (类似 tweet_freq)
      2: len(screen_name)
      3: friends_count
      4: statuses_count
      5: interactions_count
      6: bigrams_freq(screen_name)
      7: entropy(screen_name)
      8: entropy(description)
    """
    rows = []
    for uid in uids:
        p = profiles.get(uid, {})
        followers = safe_int(p.get("followers_count"))
        friends   = safe_int(p.get("friends_count"))
        statuses  = safe_int(p.get("statuses_count"))
        interact  = safe_int(p.get("interactions_count"))
        screen    = p.get("screen_name") or ""
        desc      = p.get("description") or ""
        created   = p.get("created_at") or ""

        active = compute_active_days(created)
        tweet_freq_val = interact / active

        rows.append([
            float(followers),
            float(tweet_freq_val),
            float(len(screen)),
            float(friends),
            float(statuses),
            float(interact),
            float(bigrams_freq(screen)),
            float(entropy(screen)),
            float(entropy(desc)),
        ])
    return np.array(rows, dtype=np.float32)  # [N, 9]


def zscore_normalize(train_arr, other_arrs):
    """用 train_arr 计算 mean/std，对所有 arrs 进行 z-score 归一化（避免 nan）"""
    mean = train_arr.mean(axis=0)
    std  = train_arr.std(axis=0)
    std[std == 0] = 1.0

    normalized = [(arr - mean) / std for arr in [train_arr] + list(other_arrs)]
    return normalized, mean, std

# ── BERT 嵌入 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_descriptions(uids, profiles, tokenizer, model, device):
    """
    用 xlm-roberta-base 提取每个用户 description 的 pooler_output (768-dim)
    返回 Tensor [N, 1, 768]
    """
    vecs = []
    for uid in tqdm(uids, desc="  BERT encoding"):
        p = profiles.get(uid, {})
        desc = (p.get("description") or "").strip()
        if not desc:
            vecs.append(torch.zeros(768))
            continue
        inputs = tokenizer(
            desc,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        vecs.append(out.pooler_output[0].cpu())  # [768]
    return torch.stack(vecs, dim=0).unsqueeze(1)  # [N, 1, 768]

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=== DeeProBot Weibo-1000 Preprocessing ===\n")

    # 1. 加载标签和 profile
    print("[1/5] Loading labels and profiles...")
    label_dict = load_labels()
    profiles   = load_profiles()

    # 2. 加载官方划分 split.json
    print("\n[1.5/5] Loading official split from split.json...")
    with open(DATA_DIR / "split.json", "r", encoding="utf-8") as f:
        split_dict = json.load(f)  # {uid: 'train'/'val'/'test'/'ood'}
    split_dict = {str(k): v for k, v in split_dict.items()}

    # 3. 按官方划分分配用户
    print("\n[2/5] Assigning users by official split...")
    train_uids, train_ys = [], []
    val_uids,   val_ys   = [], []
    test_uids,  test_ys  = [], []
    agent_uids           = []   # ood 集合，ground truth = bot

    for uid, split_val in split_dict.items():
        if uid not in profiles:
            continue  # 无 profile 则跳过
        lbl = label_dict.get(uid)
        if split_val == "ood":
            agent_uids.append(uid)
        elif split_val in ("train", "val", "test"):
            if lbl == 0:
                y = 0
            elif lbl == 2:
                y = 1
            else:
                continue  # 标签异常则跳过
            if split_val == "train":
                train_uids.append(uid); train_ys.append(y)
            elif split_val == "val":
                val_uids.append(uid);   val_ys.append(y)
            else:
                test_uids.append(uid);  test_ys.append(y)

    print(f"  train={len(train_uids)} (human={train_ys.count(0)}, bot={train_ys.count(1)})")
    print(f"  val  ={len(val_uids)}   (human={val_ys.count(0)}, bot={val_ys.count(1)})")
    print(f"  test ={len(test_uids)}  (human={test_ys.count(0)}, bot={test_ys.count(1)})")
    print(f"  agent (ood)={len(agent_uids)}")

    # 4. 提取数值特征
    print("\n[3/5] Extracting numerical features...")
    train_num = extract_num_features(train_uids, profiles)
    val_num   = extract_num_features(val_uids,   profiles)
    test_num  = extract_num_features(test_uids,  profiles)
    agent_num = extract_num_features(agent_uids, profiles)

    # z-score：用训练集的 mean/std
    [train_num_norm, val_num_norm, test_num_norm, agent_num_norm], norm_mean, norm_std = \
        zscore_normalize(train_num, [val_num, test_num, agent_num])

    # 重组（labeled = train + val + test 的 concat，顺序固定）
    all_labeled_uids = train_uids + val_uids + test_uids
    all_num_norm = np.concatenate([train_num_norm, val_num_norm, test_num_norm], axis=0)

    # 5. BERT 编码
    print("\n[4/5] Encoding descriptions with xlm-roberta-base...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_PATH))
    bert_model = AutoModel.from_pretrained(str(BERT_PATH)).to(device).eval()

    print("  Encoding labeled users...")
    labeled_des = encode_descriptions(all_labeled_uids, profiles, tokenizer, bert_model, device)
    print("  Encoding agent users...")
    agent_des   = encode_descriptions(agent_uids, profiles, tokenizer, bert_model, device)

    del bert_model  # 释放显存/内存

    # 确保张量在 CPU 上（MPS/CUDA 张量无法直接 torch.save 到正常路径）
    labeled_des = labeled_des.cpu()
    agent_des   = agent_des.cpu()

    # 6. 构建 labels tensor
    all_labeled_ys = train_ys + val_ys + test_ys
    labels_tensor = torch.tensor(all_labeled_ys, dtype=torch.long)

    # 7. 索引张量（相对于 all_labeled_uids 的位置）
    N_train = len(train_uids)
    N_val   = len(val_uids)
    N_test  = len(test_uids)
    train_idx = torch.arange(0,                  N_train,           dtype=torch.long)
    val_idx   = torch.arange(N_train,             N_train + N_val,   dtype=torch.long)
    test_idx  = torch.arange(N_train + N_val,     N_train + N_val + N_test, dtype=torch.long)

    # 8. 保存
    print("\n[5/5] Saving tensors to processed/...")
    torch.save(labeled_des,                          SAVE_DIR / "des_tensor.pt")
    torch.save(torch.tensor(all_num_norm),           SAVE_DIR / "num_prop.pt")
    torch.save(labels_tensor,                        SAVE_DIR / "labels.pt")
    torch.save(train_idx,                            SAVE_DIR / "train_idx.pt")
    torch.save(val_idx,                              SAVE_DIR / "val_idx.pt")
    torch.save(test_idx,                             SAVE_DIR / "test_idx.pt")
    torch.save(agent_des,                            SAVE_DIR / "agent_des.pt")
    torch.save(torch.tensor(agent_num_norm),         SAVE_DIR / "agent_num_prop.pt")
    torch.save({"mean": torch.tensor(norm_mean),
                "std":  torch.tensor(norm_std)},     SAVE_DIR / "norm_stats.pt")

    # 保存 uid 列表（供调试用）
    with open(SAVE_DIR / "labeled_uids.json", "w", encoding="utf-8") as f:
        json.dump(all_labeled_uids, f, ensure_ascii=False)
    with open(SAVE_DIR / "agent_uids.json", "w", encoding="utf-8") as f:
        json.dump(agent_uids, f, ensure_ascii=False)

    print("\n=== Preprocessing Complete ===")
    print(f"  des_tensor:    {labeled_des.shape}")
    print(f"  num_prop:      {all_num_norm.shape}")
    print(f"  labels:        {labels_tensor.shape}")
    print(f"  train/val/test idx: {train_idx.shape[0]} / {val_idx.shape[0]} / {test_idx.shape[0]}")
    print(f"  agent_des:     {agent_des.shape}")
    print(f"  agent_num:     {agent_num_norm.shape}")


if __name__ == "__main__":
    main()
