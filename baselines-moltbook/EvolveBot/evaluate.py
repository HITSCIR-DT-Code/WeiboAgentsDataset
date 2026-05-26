"""
  cd baselines-moltbook/EvolveBot
  python preprocess.py

  python evaluate.py --checkpoint ../../baselines-weibo/EvolveBot/checkpoints/rf_nograph_seed42.joblib


  python evaluate.py \\
      --checkpoint ../../baselines-weibo/EvolveBot/checkpoints/rf_nograph_seed42.joblib \\
      --processed_dir ./processed
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


BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EvolveBot checkpoint on Moltbook OOD data")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="已训练的 EvolveBot .joblib checkpoint 路径",
    )
    parser.add_argument(
        "--processed_dir", type=str, default=str(BASE_DIR / "processed"),
        help="Moltbook 预处理输出目录（默认 ./processed）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall":    recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1":        f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def print_metrics(title: str, metrics: dict):
    print(title)
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  auc      : {metrics['auc']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    proc_dir  = Path(args.processed_dir)

    print("=== EvolveBot Moltbook OOD Evaluation ===")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Processed dir: {proc_dir}")

    # ----- load checkpoint -----
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = joblib.load(ckpt_path)
    clf           = ckpt["model"]
    feature_set   = ckpt.get("feature_set", "unknown")
    feature_names = ckpt.get("feature_names", [])
    random_seed   = ckpt.get("random_seed", "?")

    if feature_set != "nograph":
        raise ValueError(
            f"Checkpoint feature_set='{feature_set}' is not supported for Moltbook OOD.\n"
            "Only 'nograph' checkpoints are supported because Moltbook lacks follow-graph data."
        )

    print(f"Feature set  : {feature_set}  ({len(feature_names)} dims)")
    print(f"Features     : {', '.join(feature_names)}")
    print(f"Random seed  : {random_seed}")

    # ----- load preprocessed features -----
    feat_path = proc_dir / "features.pt"
    user_path = proc_dir / "user_list.json"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found: {feat_path}\n"
            "Run preprocess.py first."
        )
    features = torch.load(feat_path, weights_only=True).cpu().numpy().astype(np.float32)
    with open(user_path, "r", encoding="utf-8") as f:
        user_list = json.load(f)

    n_users = len(user_list)
    print(f"Moltbook users: {n_users}")
    print(f"Feature shape : {features.shape}")

    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dim mismatch: checkpoint expects {len(feature_names)} dims, "
            f"but processed features have {features.shape[1]} dims."
        )

    # ----- inference -----
    y_pred = clf.predict(features)
    y_prob = clf.predict_proba(features)[:, 1]
    y_true = np.ones(n_users, dtype=np.int64)   # all Moltbook users are bots

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(f"\nOOD Results on Moltbook (all ground truth = bot, N={n_users})", metrics)

    # ----- per-class prediction summary -----
    n_bot_pred   = int((y_pred == 1).sum())
    n_human_pred = int((y_pred == 0).sum())
    print(f"\nPrediction breakdown:")
    print(f"  Predicted bot  : {n_bot_pred}  ({n_bot_pred / n_users * 100:.1f}%)")
    print(f"  Predicted human: {n_human_pred} ({n_human_pred / n_users * 100:.1f}%)")


if __name__ == "__main__":
    main()
