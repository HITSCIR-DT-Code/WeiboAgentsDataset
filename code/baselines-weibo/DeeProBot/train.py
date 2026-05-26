import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from model import DeeProBot

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
PROC_DIR  = BASE_DIR / "processed"
CKPT_DIR  = BASE_DIR / "checkpoints"


# ── 评估函数 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
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

    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0, average="macro")
    precision = precision_score(all_labels, all_preds, zero_division=0, average="macro")
    recall    = recall_score(all_labels, all_preds, zero_division=0, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return acc, f1, precision, recall, auc


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(args):
    # 为本次训练生成唯一 checkpoint 路径（时间戳）
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = CKPT_DIR / f"best_model_{run_id}.pt"
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device in ("cuda", "mps"):
        # 请求的加速器不可用，回退到 cpu
        print(f"  Warning: {args.device} not available, falling back to cpu.")
        device = torch.device("cpu")
    else:
        # 自动检测： cuda > mps > cpu
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. 加载张量
    print("\n[1/3] Loading preprocessed tensors...")
    des_tensor    = torch.load(PROC_DIR / "des_tensor.pt",    weights_only=True)  # [N, 1, 768]
    num_prop      = torch.load(PROC_DIR / "num_prop.pt",      weights_only=True).float()  # [N, 9]
    labels        = torch.load(PROC_DIR / "labels.pt",        weights_only=True)  # [N]
    train_idx     = torch.load(PROC_DIR / "train_idx.pt",     weights_only=True)
    val_idx       = torch.load(PROC_DIR / "val_idx.pt",       weights_only=True)
    test_idx      = torch.load(PROC_DIR / "test_idx.pt",      weights_only=True)
    agent_des     = torch.load(PROC_DIR / "agent_des.pt",     weights_only=True)  # [M, 1, 768]
    agent_num     = torch.load(PROC_DIR / "agent_num_prop.pt",weights_only=True).float()  # [M, 9]

    def make_loader(idx, shuffle=False):
        ds = TensorDataset(des_tensor[idx], num_prop[idx], labels[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    n_train = len(train_idx)
    n_bot   = labels[train_idx].sum().item()
    n_human = n_train - n_bot
    print(f"  train={n_train} (human={int(n_human)}, bot={int(n_bot)}), "
          f"val={len(val_idx)}, test={len(test_idx)}, agent={len(agent_des)}")

    # 2. 模型 & 优化器
    print("\n[2/3] Training...")
    model = DeeProBot(des_size=768, num_prop_size=9, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.weighted_loss:
        train_labels_cpu = labels[train_idx].cpu()
        counts = torch.bincount(train_labels_cpu, minlength=2).float()
        class_weight = counts.sum() / (counts.clamp_min(1.0) * 2)
        class_weight = class_weight.to(device)
        print(f"  [weighted_loss] class weights: human={class_weight[0]:.4f}, bot={class_weight[1]:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1  = -1.0
    best_epoch   = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for des, num, lbl in train_loader:
            des, num, lbl = des.to(device), num.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(des, num)
            loss   = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lbl)

        avg_loss = total_loss / n_train
        val_acc, val_f1, val_prec, val_rec, val_auc = evaluate(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                  f"val_prec={val_prec:.4f}  val_rec={val_rec:.4f}  val_auc={val_auc:.4f}"
                  f"  {'← best' if epoch == best_epoch else ''}")

    print(f"\n  Best model at epoch {best_epoch}, val_f1={best_val_f1:.4f}, saved to {ckpt_path}")

    # 3. 测试集评估
    print("\n[3/3] Evaluation...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    test_acc, test_f1, test_prec, test_rec, test_auc = evaluate(model, test_loader, device)
    print(f"\n  ── Test Set ──")
    print(f"  Accuracy       : {test_acc:.4f}")
    print(f"  F1             : {test_f1:.4f}")
    print(f"  Precision      : {test_prec:.4f}")
    print(f"  Recall         : {test_rec:.4f}")
    print(f"  AUC            : {test_auc:.4f}")

    # OOD 评估（agent 账户，ground truth = 1）
    if len(agent_des) > 0:
        agent_ds     = TensorDataset(agent_des, agent_num,
                                     torch.ones(len(agent_des), dtype=torch.long))
        agent_loader = DataLoader(agent_ds, batch_size=args.batch_size, shuffle=False)
        ood_acc, ood_f1, ood_prec, ood_rec, ood_auc = evaluate(model, agent_loader, device)
        print(f"\n  ── OOD (Agent Accounts, ground truth=bot) ──")
        print(f"  Accuracy       : {ood_acc:.4f}")
        print(f"  F1             : {ood_f1:.4f}")
        print(f"  Precision      : {ood_prec:.4f}")
        print(f"  Recall         : {ood_rec:.4f}")
        print(f"  AUC            : {ood_auc:.4f}")
    else:
        print("\n  No agent accounts found, skipping OOD evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeeProBot on Weibo-1000")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--device",     type=str,   default="auto",
                        help="cuda / mps / cpu / auto (default: auto)")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="启用后使用训练集类别频率的倒数对 CrossEntropyLoss 加权，以缓解类别不平衡")
    args = parser.parse_args()
    main(args)
