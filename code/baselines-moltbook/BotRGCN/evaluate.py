"""
  cd baselines-moltbook/BotRGCN

  python evaluate.py

  python evaluate.py --checkpoint ../../baselines-weibo/BotRGCN/checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import BotRGCN

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_CHECKPOINT = (
    ROOT_DIR / "baselines-weibo" / "BotRGCN" / "checkpoints" / "best_model.pt"
)
DEFAULT_PROCESSED_DIR = (
    BASE_DIR.parent / "moltbook-preprocess" / "moltbook_processed_data"
)


# ── 设备检测 ──────────────────────────────────────────────────────────────────
def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="BotRGCN OOD evaluation on Moltbook dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"已训练的 BotRGCN checkpoint 路径（.pt）\n（默认：{DEFAULT_CHECKPOINT}）",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help=f"Moltbook 预处理输出目录\n（默认：{DEFAULT_PROCESSED_DIR}）",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="BotRGCN embedding dimension，须与训练时一致（默认：128）",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout 比率，须与 BotRGCN 模型定义一致（默认：0.3，推断时不生效）",
    )
    return parser.parse_args()


# ── 双向边增广（与 train.py 完全一致）────────────────────────────────────────
def ensure_bidirectional_relations(
    edge_index_tensor: torch.Tensor,
    edge_type_tensor: torch.Tensor,
):
    """
    若 edge_type 只有一种取值，则为原始边（type=0）补充反向边（type=1），
    以满足 RGCNConv num_relations=2 的要求。
    逻辑与 baselines-weibo/BotRGCN/train.py 完全相同。
    """
    unique_types = torch.unique(edge_type_tensor)
    if unique_types.numel() > 1:
        return edge_index_tensor, edge_type_tensor

    reverse_edge_index = torch.stack(
        (edge_index_tensor[1], edge_index_tensor[0]), dim=0
    )
    reverse_edge_type = torch.ones(
        reverse_edge_index.shape[1],
        dtype=edge_type_tensor.dtype,
        device=edge_type_tensor.device,
    )
    augmented_edge_index = torch.cat((edge_index_tensor, reverse_edge_index), dim=1)
    augmented_edge_type = torch.cat((edge_type_tensor, reverse_edge_type), dim=0)
    return augmented_edge_index, augmented_edge_type


# ── 评估指标 ──────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall":    recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1":        f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = float("nan")
    else:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(title)
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  auc      : {metrics['auc']:.4f}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    proc_dir = Path(args.processed_dir)
    device = detect_device()

    print("=== BotRGCN Moltbook OOD Evaluation ===")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Processed dir: {proc_dir}")
    print(f"Device       : {device}")

    # ── [1] 加载检查点 ────────────────────────────────────────────────────────
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "请先运行 baselines-weibo/BotRGCN/train.py 训练模型，"
            "或通过 --checkpoint 指定正确的检查点路径。"
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(
        f"\nLoaded checkpoint:"
        f"\n  epoch  : {checkpoint.get('epoch', 'N/A')}"
        f"\n  val_f1 : {checkpoint.get('val_f1', float('nan')):.4f}"
        f"\n  weighted_loss: {checkpoint.get('weighted_loss', False)}"
    )

    # ── [2] 加载预处理张量 ────────────────────────────────────────────────────
    required_files = [
        "des_tensor.pt",
        "tweets_tensor.pt",
        "num_properties_tensor.pt",
        "cat_properties_tensor.pt",
        "edge_index.pt",
        "edge_type.pt",
    ]
    for fname in required_files:
        fpath = proc_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"预处理文件缺失: {fpath}\n"
                "请先运行：\n"
                "  cd baselines-moltbook/moltbook-preprocess\n"
                "  python preprocess_moltbook.py"
            )

    def load(filename: str) -> torch.Tensor:
        return torch.load(proc_dir / filename, map_location=device)

    des_tensor  = load("des_tensor.pt")
    tweets_tensor = load("tweets_tensor.pt")
    num_prop    = load("num_properties_tensor.pt")
    cat_prop    = load("cat_properties_tensor.pt")
    edge_index  = load("edge_index.pt")
    edge_type   = load("edge_type.pt")

    N = des_tensor.shape[0]
    print(f"\nMoltbook users: {N}")
    print(f"  des_tensor      : {tuple(des_tensor.shape)}")
    print(f"  tweets_tensor   : {tuple(tweets_tensor.shape)}")
    print(f"  num_prop        : {tuple(num_prop.shape)}")
    print(f"  cat_prop        : {tuple(cat_prop.shape)}")
    print(f"  edge_index      : {tuple(edge_index.shape)}")
    print(f"  edge_type       : {tuple(edge_type.shape)}")

    # ── [3] 双向边增广 ────────────────────────────────────────────────────────
    edge_index, edge_type = ensure_bidirectional_relations(edge_index, edge_type)
    print(f"  edge_index (bidirectional): {tuple(edge_index.shape)}")

    # ── [4] 加载用户列表（可选，用于输出预测明细）────────────────────────────
    uid_path = proc_dir / "all_uids.json"
    if uid_path.exists():
        with open(uid_path, "r", encoding="utf-8") as f:
            all_uids = json.load(f)
    else:
        all_uids = [str(i) for i in range(N)]

    # ── [5] 实例化模型并加载权重 ──────────────────────────────────────────────
    model = BotRGCN(
        des_size=des_tensor.shape[1],
        tweet_size=tweets_tensor.shape[1],
        num_prop_size=num_prop.shape[1],
        cat_prop_size=cat_prop.shape[1],
        embedding_dimension=args.embedding_size,
        dropout=args.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── [6] 推断 ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(des_tensor, tweets_tensor, num_prop, cat_prop, edge_index, edge_type)

    probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predictions   = logits.argmax(dim=1).cpu().numpy()

    # Moltbook 所有账户均为 agent（机器人），真实标签全为 1
    y_true = np.ones(N, dtype=np.int64)

    # ── [7] 输出评估指标 ──────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, predictions, probabilities)
    print_metrics(
        f"\nOOD Results on Moltbook (all ground truth = bot, N={N})", metrics
    )

    # ── [8] 预测分布统计 ──────────────────────────────────────────────────────
    n_bot_pred   = int((predictions == 1).sum())
    n_human_pred = int((predictions == 0).sum())
    print(f"\nPrediction breakdown:")
    print(f"  Predicted bot   (1): {n_bot_pred}  ({n_bot_pred / N * 100:.1f}%)")
    print(f"  Predicted human (0): {n_human_pred} ({n_human_pred / N * 100:.1f}%)")

    if n_human_pred > 0 and n_human_pred <= 50:
        human_predicted = [all_uids[i] for i, p in enumerate(predictions) if p == 0]
        print(f"\nUsers predicted as human: {human_predicted}")


if __name__ == "__main__":
    main()
