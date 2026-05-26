import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NOGRAPH_INDICES = [0, 2, 3, 4, 5, 6, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="Train EvolveBot on Weibo")
    parser.add_argument("random_seed", type=int, nargs="?", default=42)
    parser.add_argument("--feature_set", choices=["full", "nograph"], default="full")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="启用后对 RandomForestClassifier 使用 class_weight='balanced'，以缓解类别不平衡")
    return parser.parse_args()


def detect_device(requested):
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return "mps" if has_mps else "cpu"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return "mps"
    return "cpu"


def load_tensor(name):
    return torch.load(PROC_DIR / name, weights_only=True)


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1": f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def print_metrics(title, metrics):
    print(title)
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  auc      : {metrics['auc']:.4f}")


def main():
    args = parse_args()
    device = detect_device(args.device)
    print("=== EvolveBot Weibo Training ===")
    print(f"Detected device: {device}")
    print("RandomForestClassifier runs on CPU; device detection is reported for consistency with preprocessing.")

    features = load_tensor("features.pt").cpu().numpy().astype(np.float32)
    labels = load_tensor("labels.pt").cpu().numpy().astype(np.int64)
    train_idx = load_tensor("train_idx.pt").cpu().numpy().astype(np.int64)
    val_idx = load_tensor("val_idx.pt").cpu().numpy().astype(np.int64)
    test_idx = load_tensor("test_idx.pt").cpu().numpy().astype(np.int64)
    agent_features = load_tensor("agent_features.pt").cpu().numpy().astype(np.float32)

    with open(PROC_DIR / "feature_names.json", "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    selected_indices = list(range(features.shape[1])) if args.feature_set == "full" else NOGRAPH_INDICES
    selected_feature_names = [feature_names[index] for index in selected_indices]
    features = features[:, selected_indices]
    agent_features = agent_features[:, selected_indices]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
        class_weight="balanced" if args.weighted_loss else None,
        n_jobs=-1,
    )
    if args.weighted_loss:
        print("[weighted_loss] 已启用 class_weight='balanced'")
    clf.fit(x_train, y_train)

    val_pred = clf.predict(x_val)
    val_prob = clf.predict_proba(x_val)[:, 1]
    test_pred = clf.predict(x_test)
    test_prob = clf.predict_proba(x_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    print(f"Feature set: {args.feature_set} ({len(selected_feature_names)} dims)")
    print(f"Features: {', '.join(selected_feature_names)}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print_metrics("\nValidation", val_metrics)
    print_metrics("\nTest", test_metrics)

    if len(agent_features) > 0:
        agent_pred = clf.predict(agent_features)
        agent_prob = clf.predict_proba(agent_features)[:, 1]
        agent_truth = np.ones(len(agent_features), dtype=np.int64)
        ood_metrics = compute_metrics(agent_truth, agent_pred, agent_prob)
        print_metrics("\nOOD (agent accounts, ground truth=bot)", ood_metrics)
    else:
        print("\nNo agent accounts found, skipping OOD evaluation.")

    checkpoint_path = MODEL_DIR / f"rf_{args.feature_set}_seed{args.random_seed}.joblib"
    joblib.dump(
        {
            "model": clf,
            "feature_set": args.feature_set,
            "feature_indices": selected_indices,
            "feature_names": selected_feature_names,
            "random_seed": args.random_seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"\nSaved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()