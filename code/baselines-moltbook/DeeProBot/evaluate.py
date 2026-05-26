"""
  cd baselines-moltbook/DeeProBot
  python evaluate.py
  python evaluate.py --checkpoint ../../baselines-weibo/DeeProBot/best_model.pt
  python evaluate.py --processed_dir processed --batch_size 128 --device cpu
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from model import DeeProBot

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DeeProBot OOD evaluation on Moltbook")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "DeeProBot" / "best_model.pt"),
        help="训练好的模型权重路径（默认：baselines-weibo/DeeProBot/best_model.pt）",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(BASE_DIR / "processed"),
        help="预处理输出目录（默认：processed/，需先运行 preprocess.py）",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / mps / cpu / auto（默认 auto）")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="推断时 dropout 概率（默认 0，即关闭 dropout）")
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


# ── 评估函数 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for des, num, labels in loader:
        des, num, labels = des.to(device), num.to(device), labels.to(device)
        logits = model(des, num)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_probs, all_labels


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = detect_device(args.device)
    proc_dir = Path(args.processed_dir)
    ckpt_path = Path(args.checkpoint)

    print("=== DeeProBot OOD Evaluation on Moltbook ===")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Processed dir: {proc_dir}")
    print(f"Device       : {device}")

    # 1. 检查预处理文件
    des_path = proc_dir / "des.pt"
    num_path = proc_dir / "num_prop.pt"
    for p in [des_path, num_path, ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"文件不存在: {p}\n"
                "请先运行 preprocess.py 生成预处理文件，并确认模型权重路径正确。"
            )

    # 2. 加载预处理张量
    print("\n[1/3] Loading preprocessed tensors...")
    des_tensor = torch.load(des_path, weights_only=True)   # [N, 1, 768]
    num_prop   = torch.load(num_path, weights_only=True).float()  # [N, 9]

    # 校验维度
    assert des_tensor.ndim == 3 and des_tensor.shape[1] == 1 and des_tensor.shape[2] == 768, \
        f"des.pt 维度异常，期望 [N, 1, 768]，实际 {des_tensor.shape}"
    assert num_prop.ndim == 2 and num_prop.shape[1] == 9, \
        f"num_prop.pt 维度异常，期望 [N, 9]，实际 {num_prop.shape}"
    assert des_tensor.shape[0] == num_prop.shape[0], \
        f"样本数不一致：des={des_tensor.shape[0]}, num={num_prop.shape[0]}"

    N = des_tensor.shape[0]
    print(f"  {N} users loaded")

    # 加载 user_list（可选，用于输出参考）
    user_list_path = proc_dir / "user_list.json"
    if user_list_path.exists():
        with open(user_list_path, "r", encoding="utf-8") as f:
            user_list = json.load(f)
        assert len(user_list) == N, \
            f"user_list.json 长度 {len(user_list)} 与张量样本数 {N} 不符"
    else:
        user_list = None

    # 3. 初始化模型并加载权重
    print("\n[2/3] Loading model checkpoint...")
    model = DeeProBot(des_size=768, num_prop_size=9, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model loaded from {ckpt_path}")

    # 4. 推断
    print("\n[3/3] Running inference...")
    # Moltbook 全部为 agent（机器人），真实标签 = 1
    labels_tensor = torch.ones(N, dtype=torch.long)
    dataset = TensorDataset(des_tensor, num_prop, labels_tensor)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds, probs, labels = run_inference(model, loader, device)

    # 5. 计算指标
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, zero_division=0, average="macro")
    precision = precision_score(labels, preds, zero_division=0, average="macro")
    recall    = recall_score(labels, preds, zero_division=0, average="macro")
    try:
        # 所有真实标签相同（全为1），AUC 无意义，使用 bot 概率分数做近似
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    n_pred_bot   = sum(1 for p in preds if p == 1)
    n_pred_human = sum(1 for p in preds if p == 0)

    print("\n" + "=" * 50)
    print("  OOD Evaluation Results (Moltbook, ground truth=bot)")
    print("=" * 50)
    print(f"  Total users      : {N}")
    print(f"  Pred as bot (1)  : {n_pred_bot}  ({n_pred_bot/N*100:.1f}%)")
    print(f"  Pred as human(0) : {n_pred_human}  ({n_pred_human/N*100:.1f}%)")
    print("-" * 50)
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1 (macro)       : {f1:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro)   : {recall:.4f}")
    print(f"  AUC              : {auc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
