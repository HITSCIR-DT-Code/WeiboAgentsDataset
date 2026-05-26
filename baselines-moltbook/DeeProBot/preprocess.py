"""
  cd baselines-moltbook/DeeProBot
  python preprocess.py
  python preprocess.py --device cpu
  python preprocess.py --weibo_processed_dir ../../baselines-weibo/DeeProBot/processed
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import bigrams_freq, entropy

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "dataset" / "Moltbook"
MODEL_DIR = ROOT_DIR / "models" / "xlm-roberta-base"
SAVE_DIR = BASE_DIR / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 账号年龄截止日期（与 Weibo 数据收集时间对齐）
REFERENCE_DATE = datetime(2026, 4, 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Moltbook data for DeeProBot OOD")
    parser.add_argument(
        "--weibo_processed_dir",
        type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "DeeProBot" / "processed"),
        help="Weibo DeeProBot processed 目录，用于读取归一化参数（norm_stats.pt）",
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / mps / cpu / auto（默认 auto）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="BERT 编码批大小（默认 32）")
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


# ── I/O 工具 ──────────────────────────────────────────────────────────────────

def iterate_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def parse_date(value: str):
    """解析 join_time，格式为 YYYY-MM-DD（仅日期精度）"""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def compute_active_days(join_time_str: str) -> float:
    """计算从 join_time 到 REFERENCE_DATE 的天数，最小为 1"""
    dt = parse_date(join_time_str)
    if dt is None:
        return 1.0
    days = (REFERENCE_DATE - dt).days
    return max(float(days), 1.0)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_user_info() -> dict:
    """返回 {author: info_dict}"""
    users = {}
    for item in iterate_jsonl(DATA_DIR / "user_info.jsonl"):
        author = item.get("author", "")
        if author:
            users[author] = item
    return users


def load_statuses_count(known_authors: set) -> dict:
    """
    流式遍历 user_history_posts.jsonl，统计每用户的帖子行数作为 statuses_count 的近似。
    注意：爬虫实际抓取量可能少于平台显示总数，此为可用的最佳近似值。
    返回 {author: count}
    """
    counts: dict[str, int] = defaultdict(int)
    for item in tqdm(iterate_jsonl(DATA_DIR / "user_history_posts.jsonl"),
                     desc="  Counting history posts", mininterval=5.0):
        author = item.get("author", "")
        if author in known_authors:
            counts[author] += 1
    return dict(counts)


# ── 特征提取 ──────────────────────────────────────────────────────────────────

def extract_num_features(ordered_authors: list, user_info: dict,
                         statuses_counts: dict) -> np.ndarray:
    """
    提取 9 维数值特征，顺序与 Weibo DeeProBot 完全一致：
      0: followers_count
      1: interactions_count / active_days  (karma / active_days)
      2: len(screen_name)                  (len(author))
      3: friends_count                     (following)
      4: statuses_count                    (history posts count)
      5: interactions_count                (karma)
      6: bigrams_freq(screen_name)         (bigrams_freq(author))
      7: entropy(screen_name)              (entropy(author))
      8: entropy(description)              (entropy(discription))
    返回 np.ndarray [N, 9]
    """
    rows = []
    for author in ordered_authors:
        info = user_info[author]

        followers  = float(safe_int(info.get("followers")))
        following  = float(safe_int(info.get("following")))
        karma      = float(safe_int(info.get("karma")))
        desc       = (info.get("discription") or "").strip()   # 注意原始字段拼写
        join_time  = info.get("join_time", "")
        sc         = float(statuses_counts.get(author, 0))

        active_days    = compute_active_days(join_time)
        karma_per_day  = karma / active_days

        rows.append([
            followers,           # 0: followers_count
            karma_per_day,       # 1: interactions_count / active_days
            float(len(author)),  # 2: len(screen_name)
            following,           # 3: friends_count
            sc,                  # 4: statuses_count
            karma,               # 5: interactions_count
            float(bigrams_freq(author)),  # 6: bigrams_freq(screen_name)
            float(entropy(author)),       # 7: entropy(screen_name)
            float(entropy(desc)),         # 8: entropy(description)
        ])
    return np.array(rows, dtype=np.float32)   # [N, 9]


# ── BERT 编码 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_descriptions(ordered_authors: list, user_info: dict,
                         tokenizer, model, device,
                         batch_size: int) -> torch.Tensor:
    """
    对每个用户的 discription 字段用 xlm-roberta-base 提取 pooler_output (768-dim)。
    返回 Tensor [N, 1, 768]（与 Weibo DeeProBot 的 des_tensor 格式一致）。
    空描述补零向量。
    """
    model.eval()
    all_vecs = []
    for i in tqdm(range(0, len(ordered_authors), batch_size), desc="  BERT encoding"):
        batch_authors = ordered_authors[i: i + batch_size]
        texts = []
        has_text = []
        for author in batch_authors:
            desc = (user_info[author].get("discription") or "").strip()
            texts.append(desc)
            has_text.append(bool(desc))

        # 对空描述填充占位符，之后再替换为零向量
        inputs_text = [t if t else "[PAD]" for t in texts]
        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        # pooler_output: [batch, 768]
        pooler = out.pooler_output.cpu()   # [batch, 768]

        for j, has in enumerate(has_text):
            if has:
                all_vecs.append(pooler[j])
            else:
                all_vecs.append(torch.zeros(768))

    return torch.stack(all_vecs, dim=0).unsqueeze(1)   # [N, 1, 768]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = detect_device(args.device)
    weibo_proc_dir = Path(args.weibo_processed_dir)

    print("=== DeeProBot Moltbook OOD Preprocessing ===")
    print(f"Data dir    : {DATA_DIR}")
    print(f"Weibo proc  : {weibo_proc_dir}")
    print(f"Model dir   : {MODEL_DIR}")
    print(f"Device      : {device}")

    # 1. 加载 Weibo 归一化参数
    print("\n[1/5] Loading normalization stats from Weibo training set...")
    norm_stats_path = weibo_proc_dir / "norm_stats.pt"
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Weibo norm_stats.pt not found: {norm_stats_path}\n"
            "请先运行 baselines-weibo/DeeProBot/preprocess.py 生成归一化参数。"
        )
    norm_stats = torch.load(norm_stats_path, weights_only=True)
    norm_mean = norm_stats["mean"].numpy().astype(np.float32)   # [9]
    norm_std  = norm_stats["std"].numpy().astype(np.float32)    # [9]
    norm_std[norm_std == 0] = 1.0
    print(f"  Loaded mean/std, shape: {norm_mean.shape}")

    # 2. 加载用户信息
    print("\n[2/5] Loading user_info.jsonl...")
    user_info = load_user_info()
    ordered_authors = sorted(user_info.keys())   # 确定性排序
    print(f"  {len(ordered_authors)} users loaded")

    # 3. 统计历史帖子数（作为 statuses_count 近似）
    print("\n[3/5] Counting history posts (statuses_count approximation)...")
    statuses_counts = load_statuses_count(set(ordered_authors))
    covered = sum(1 for a in ordered_authors if a in statuses_counts)
    print(f"  {covered}/{len(ordered_authors)} users have history posts")

    # 4. 提取并归一化数值特征
    print("\n[4/5] Extracting numerical features...")
    raw_arr = extract_num_features(ordered_authors, user_info, statuses_counts)  # [N, 9]
    norm_arr = (raw_arr - norm_mean) / norm_std                                  # [N, 9]
    print(f"  Feature shape: {raw_arr.shape}")

    # 5. BERT 编码
    print("\n[5/5] Encoding descriptions with xlm-roberta-base...")
    tokenizer  = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    bert_model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device).eval()
    des_tensor = encode_descriptions(
        ordered_authors, user_info, tokenizer, bert_model, device, args.batch_size
    )
    del bert_model   # 释放显存
    des_tensor = des_tensor.cpu()
    print(f"  Description tensor shape: {des_tensor.shape}")

    # 保存
    torch.save(des_tensor,                              SAVE_DIR / "des.pt")
    torch.save(torch.tensor(norm_arr, dtype=torch.float32), SAVE_DIR / "num_prop.pt")
    torch.save(torch.tensor(raw_arr,  dtype=torch.float32), SAVE_DIR / "raw_num_prop.pt")
    with open(SAVE_DIR / "user_list.json", "w", encoding="utf-8") as f:
        json.dump(ordered_authors, f, ensure_ascii=False, indent=2)

    print(f"\n=== Preprocessing Complete ===")
    print(f"  Users        : {len(ordered_authors)}")
    print(f"  des.pt       : {des_tensor.shape}")
    print(f"  num_prop.pt  : {norm_arr.shape}")
    print(f"  Saved to     : {SAVE_DIR}")


if __name__ == "__main__":
    main()
