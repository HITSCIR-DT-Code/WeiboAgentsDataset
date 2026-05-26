"""
  python train.py [--path baselines/RGT/processed/] [--epochs 50] [--random-seed 42]
"""

import argparse
import json
import os
from datetime import datetime
from os import makedirs
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from layer import RGTLayer

# ── 设备检测 ──────────────────────────────────────────────────────────────────
def detect_device(allow_mps_fallback=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if allow_mps_fallback:
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                print("[Warning] --allow-mps-fallback is set, but PYTORCH_ENABLE_MPS_FALLBACK=1 is missing. Unsupported ops may still fail on MPS.")
            else:
                print("[Info] Using MPS with PYTORCH_ENABLE_MPS_FALLBACK=1. Unsupported ops will fall back to CPU and may run slower than pure CPU.")
            return torch.device("mps")
        print("[Info] MPS is available but disabled for RGT because PyG sampling and TransformerConv rely on ops not implemented on MPS; use --allow-mps-fallback with PYTORCH_ENABLE_MPS_FALLBACK=1 to force MPS fallback, or stay on CPU.")
    return torch.device("cpu")


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_data(path):
    def load(name):
        return torch.load(path / name, map_location="cpu")

    cat_features    = load("cat_properties_tensor.pt")   # [N, 2]
    prop_features   = load("num_properties_tensor.pt")   # [N, 5]
    tweet_features  = load("tweets_tensor.pt")           # [N, 768]
    des_features    = load("des_tensor.pt")              # [N, 768]
    edge_index      = load("edge_index.pt")              # [2, E]
    edge_type       = load("edge_type.pt").unsqueeze(-1) # [E, 1]
    label           = load("label.pt")                   # [N]
    train_idx       = load("train_idx.pt")
    val_idx         = load("val_idx.pt")
    test_idx        = load("test_idx.pt")
    agent_idx       = load("agent_idx.pt")

    # 拼接特征：cat | num | tweet | des
    x = torch.cat([cat_features, prop_features, tweet_features, des_features], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=label)
    data.train_idx = train_idx
    data.val_idx   = val_idx
    data.test_idx  = test_idx
    data.agent_idx = agent_idx

    return data, cat_features.shape[1], prop_features.shape[1], \
           tweet_features.shape[1], des_features.shape[1]


# ── 模型 ──────────────────────────────────────────────────────────────────────
class RGTDetector(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cat_num      = args.cat_num
        self.numeric_num  = args.numeric_num
        self.tweet_channel = args.tweet_channel
        self.des_channel  = args.des_channel

        lin4 = args.linear_channels // 4

        self.in_linear_numeric = nn.Linear(args.numeric_num,   lin4, bias=True)
        self.in_linear_bool    = nn.Linear(args.cat_num,       lin4, bias=True)
        self.in_linear_tweet   = nn.Linear(args.tweet_channel, lin4, bias=True)
        self.in_linear_des     = nn.Linear(args.des_channel,   lin4, bias=True)
        self.linear1           = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(
            num_edge_type=2,
            in_channel=args.linear_channels,
            out_channel=args.out_channel,
            trans_heads=args.trans_head,
            semantic_head=args.semantic_head,
            dropout=args.dropout,
        )
        self.RGT_layer2 = RGTLayer(
            num_edge_type=2,
            in_channel=args.linear_channels,
            out_channel=args.out_channel,
            trans_heads=args.trans_head,
            semantic_head=args.semantic_head,
            dropout=args.dropout,
        )

        self.out1 = nn.Linear(args.out_channel, 64)
        self.out2 = nn.Linear(64, 2)

        self.drop  = nn.Dropout(args.dropout)
        self.ReLU  = nn.LeakyReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch):
        x          = batch.x
        edge_index = batch.edge_index
        edge_type  = batch.edge_attr.view(-1)

        cut0 = self.cat_num
        cut1 = cut0 + self.numeric_num
        cut2 = cut1 + self.tweet_channel

        cat_x   = x[:, :cut0]
        num_x   = x[:, cut0:cut1]
        tweet_x = x[:, cut1:cut2]
        des_x   = x[:, cut2:]

        feat_num   = self.drop(self.ReLU(self.in_linear_numeric(num_x)))
        feat_bool  = self.drop(self.ReLU(self.in_linear_bool(cat_x)))
        feat_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_x)))
        feat_des   = self.drop(self.ReLU(self.in_linear_des(des_x)))

        user_feat = torch.cat([feat_num, feat_bool, feat_tweet, feat_des], dim=1)
        user_feat = self.drop(self.ReLU(self.linear1(user_feat)))

        user_feat = self.ReLU(self.RGT_layer1(user_feat, edge_index, edge_type))
        user_feat = self.ReLU(self.RGT_layer2(user_feat, edge_index, edge_type))

        user_feat = self.drop(self.ReLU(self.out1(user_feat)))
        return self.out2(user_feat)


# ── 评估辅助 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, n_input_nodes):
    """对 loader 内全部 mini-batch 做推断，返回 (preds_np, probs_np, labels_np)"""
    model.eval()
    preds_list, probs_list, labels_list = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)
        # NeighborLoader 的 seed nodes 在 batch 的前 batch_size 个节点
        n_seed = batch.batch_size
        out_seed   = out[:n_seed]
        label_seed = batch.y[:n_seed]

        probs  = torch.softmax(out_seed, dim=1)
        preds  = torch.argmax(out_seed, dim=1)

        preds_list.append(preds.cpu())
        probs_list.append(probs[:, 1].cpu())
        labels_list.append(label_seed.cpu())

    preds_np  = torch.cat(preds_list).numpy()
    probs_np  = torch.cat(probs_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    return preds_np, probs_np, labels_np


def print_metrics(prefix, preds, probs, labels):
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0, average="macro")
    prec = precision_score(labels, preds, zero_division=0, average="macro")
    rec  = recall_score(labels, preds, zero_division=0, average="macro")
    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = float("nan")
    print(
        f"{prefix}  acc={acc:.4f}  f1={f1:.4f}  "
        f"precision={prec:.4f}  recall={rec:.4f}  auc={auc:.4f}"
    )
    return {"acc": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc}


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    device = detect_device(args.allow_mps_fallback)
    print(f"Device: {device}")

    # ── 数据 ─────────────────────────────────────────────────────────────────
    path = Path(args.path)
    print(f"Loading data from {path} ...")
    data, cat_num, numeric_num, tweet_channel, des_channel = load_data(path)

    # 验证参数是否与数据一致
    if args.cat_num != cat_num:
        print(f"[Warning] --cat-num={args.cat_num} but data has cat_num={cat_num}; using {cat_num}")
        args.cat_num = cat_num
    if args.numeric_num != numeric_num:
        print(f"[Warning] --numeric-num={args.numeric_num} but data has numeric_num={numeric_num}; using {numeric_num}")
        args.numeric_num = numeric_num
    if args.tweet_channel != tweet_channel:
        print(f"[Warning] --tweet-channel={args.tweet_channel} but data has {tweet_channel}; using {tweet_channel}")
        args.tweet_channel = tweet_channel
    if args.des_channel != des_channel:
        print(f"[Warning] --des-channel={args.des_channel} but data has {des_channel}; using {des_channel}")
        args.des_channel = des_channel

    n_train = data.train_idx.shape[0]
    n_val   = data.val_idx.shape[0]
    n_test  = data.test_idx.shape[0]
    n_agent = data.agent_idx.shape[0]
    print(f"Nodes: train={n_train}, val={n_val}, test={n_test}, agent(OOD)={n_agent}")
    print(f"Edges: {data.edge_index.shape[1]}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        num_neighbors=[20],
        batch_size=args.batch_size,
        num_workers=0,
    )
    train_loader = NeighborLoader(data, input_nodes=data.train_idx, shuffle=True,  **loader_kwargs)
    val_loader   = NeighborLoader(data, input_nodes=data.val_idx,   shuffle=False, **loader_kwargs)
    test_loader  = NeighborLoader(data, input_nodes=data.test_idx,  shuffle=False, **loader_kwargs)
    agent_loader = NeighborLoader(data, input_nodes=data.agent_idx, shuffle=False, **loader_kwargs)

    # ── 模型 / 优化器 ─────────────────────────────────────────────────────────
    model = RGTDetector(args).to(device)
    if args.weighted_loss:
        train_labels_cpu = data.y[data.train_idx].cpu()
        counts = torch.bincount(train_labels_cpu, minlength=2).float()
        class_weight = counts.sum() / (counts.clamp_min(1.0) * 2)
        class_weight = class_weight.to(device)
        print(f"[weighted_loss] class weights: human={class_weight[0]:.4f}, bot={class_weight[1]:.4f}")
        criterion  = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler  = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    makedirs(ckpt_dir, exist_ok=True)
    run_id=datetime.now().strftime("%Y%m%d_%H%M%S")
    best_ckpt = ckpt_dir / f"best_model_{run_id}.pt"
    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            n_seed = batch.batch_size
            out    = model(batch)
            loss   = criterion(out[:n_seed], batch.y[:n_seed])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * n_seed
            total_samples += n_seed

        scheduler.step()

        # 验证
        preds, probs, labels = evaluate(model, val_loader, device, n_val)
        val_acc = accuracy_score(labels, preds)
        val_f1  = f1_score(labels, preds, zero_division=0, average="macro")

        avg_loss = total_loss / max(total_samples, 1)
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  => Saved best model (val_f1={best_val_f1:.4f}) to {best_ckpt}")

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint from {best_ckpt} ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print("\n=== In-distribution Test Results ===")
    preds, probs, labels = evaluate(model, test_loader, device, n_test)
    print_metrics("[Test]", preds, probs, labels)

    # ── OOD（agent）评估 ──────────────────────────────────────────────────────
    if n_agent > 0:
        print(f"\n=== OOD Evaluation (agent nodes, ground truth = bot=1, n={n_agent}) ===")
        preds_ood, probs_ood, _ = evaluate(model, agent_loader, device, n_agent)
        gt_ood = torch.ones(n_agent, dtype=torch.long).numpy()
        print_metrics("[OOD]", preds_ood, probs_ood, gt_ood)
    else:
        print("\n[OOD] No agent nodes found, skipping OOD evaluation.")


# ── ArgParse ──────────────────────────────────────────────────────────────────
def parse_args():
    BASE = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="RGT Bot Detection on Weibo"
    )
    parser.add_argument("--path",          type=str,   default=str(BASE / "processed"),
                        help="预处理输出目录（包含 .pt 文件）")
    parser.add_argument("--numeric-num",   type=int,   default=5,   dest="numeric_num")
    parser.add_argument("--cat-num",       type=int,   default=2,   dest="cat_num")
    parser.add_argument("--linear-channels", type=int, default=128, dest="linear_channels")
    parser.add_argument("--des-channel",   type=int,   default=768, dest="des_channel")
    parser.add_argument("--tweet-channel", type=int,   default=768, dest="tweet_channel")
    parser.add_argument("--out-channel",   type=int,   default=128, dest="out_channel")
    parser.add_argument("--dropout",       type=float, default=0.5)
    parser.add_argument("--trans-head",    type=int,   default=8,   dest="trans_head")
    parser.add_argument("--semantic-head", type=int,   default=8,   dest="semantic_head")
    parser.add_argument("--batch-size",    type=int,   default=128, dest="batch_size")
    parser.add_argument("--epochs",        type=int,   default=300)
    parser.add_argument("--lr",            type=float, default=5e-3)
    parser.add_argument("--l2-reg",        type=float, default=1e-3, dest="l2_reg")
    parser.add_argument("--random-seed",   type=int,   default=None, dest="random_seed")
    parser.add_argument("--weighted-loss",  action="store_true", dest="weighted_loss",
                        help="启用后使用训练集类别频率的倒数对 CrossEntropyLoss 加权，以缓解类别不平衡")
    parser.add_argument("--allow-mps-fallback", action="store_true", dest="allow_mps_fallback",
                        help="在 Apple Silicon 上强制使用 MPS，并依赖 PYTORCH_ENABLE_MPS_FALLBACK=1 将未实现算子回退到 CPU")
    return parser.parse_args()


if __name__ == "__main__":
    main()
