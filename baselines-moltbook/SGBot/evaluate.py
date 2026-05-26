"""
  cd baselines-moltbook/SGBot
  python evaluate.py
  python evaluate.py --checkpoint ../../baselines-weibo/SGBot/checkpoints/rf_seed42.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SGBot OOD evaluation on Moltbook")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "baselines-weibo" / "SGBot" / "checkpoints" / "rf_seed42.joblib"),
        help="训练好的随机森林 checkpoint 路径（.joblib）"
             "（默认：baselines-weibo/SGBot/checkpoints/rf_seed42.joblib）",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(BASE_DIR / "processed"),
        help="预处理输出目录（默认：processed/，需先运行 preprocess.py）",
    )
    return parser.parse_args()


# ── 评估指标 ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall":    recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1":        f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = float("nan")
    else:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
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
    proc_dir  = Path(args.processed_dir)
    ckpt_path = Path(args.checkpoint)

    print("=== SGBot OOD Evaluation on Moltbook ===")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Processed dir: {proc_dir}")

    # 1. 加载 checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "请先运行 baselines-weibo/SGBot/train.py 训练模型，或通过 --checkpoint 指定正确路径。"
        )
    ckpt = joblib.load(ckpt_path)
    classifier   = ckpt["model"]
    feature_names = ckpt.get("feature_names", [])
    val_metrics  = ckpt.get("val_metrics", {})
    test_metrics = ckpt.get("test_metrics", {})

    print(f"\nLoaded checkpoint:")
    print(f"  random_seed  : {ckpt.get('random_seed', 'N/A')}")
    print(f"  n_estimators : {ckpt.get('n_estimators', 'N/A')}")
    print(f"  max_depth    : {ckpt.get('max_depth', 'N/A')}")
    if val_metrics:
        print(f"  Weibo val  F1: {val_metrics.get('f1', float('nan')):.4f}")
    if test_metrics:
        print(f"  Weibo test F1: {test_metrics.get('f1', float('nan')):.4f}")

    # 2. 加载预处理特征
    features_path = proc_dir / "features.pt"
    user_list_path = proc_dir / "user_list.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"features.pt not found: {features_path}\n"
            "请先运行 baselines-moltbook/SGBot/preprocess.py 生成特征。"
        )
    features = torch.load(features_path, weights_only=True).cpu().numpy().astype(np.float32)

    with open(user_list_path, "r", encoding="utf-8") as f:
        user_list = json.load(f)

    print(f"\nLoaded features: {features.shape}  ({len(user_list)} users)")

    # 验证特征维度一致性
    expected_dim = len(feature_names)
    if expected_dim > 0 and features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: checkpoint expects {expected_dim} dims, "
            f"but features.pt has {features.shape[1]} dims.\n"
            "请确认 preprocess.py 和 Weibo SGBot 的特征列表完全一致。"
        )

    # 3. 推断（Moltbook 所有账户真实标签均为 1 = bot）
    y_true = np.ones(len(user_list), dtype=np.int64)
    y_pred = classifier.predict(features)
    y_prob = classifier.predict_proba(features)[:, 1]

    # 4. 输出指标
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics("\nOOD Evaluation (Moltbook, ground truth=bot)", metrics)

    # 5. 打印被预测为 human（0）的用户列表
    human_predicted = [user_list[i] for i, p in enumerate(y_pred) if p == 0]
    bot_predicted   = [user_list[i] for i, p in enumerate(y_pred) if p == 1]

    print(f"\nPrediction summary:")
    print(f"  Total users     : {len(user_list)}")
    print(f"  Predicted as bot (1): {len(bot_predicted)}")
    print(f"  Predicted as human (0): {len(human_predicted)}")


if __name__ == "__main__":
    main()
