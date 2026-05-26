"""
python evaluate.py --checkpoint ../../baselines-weibo/RGT/checkpoints/best_checkpoint_f10.6327.pt
"""

import argparse
import sys
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
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# 将 baselines-weibo/RGT 加入路径，复用 layer.py（RGTLayer 定义）
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
WEIBO_RGT_DIR = ROOT_DIR / "baselines-weibo" / "RGT"
sys.path.insert(0, str(WEIBO_RGT_DIR))

from layer import RGTLayer  # noqa: E402 — 来自 baselines-weibo/RGT/layer.py

DEFAULT_CHECKPOINT_DIR = WEIBO_RGT_DIR / "checkpoints"
DEFAULT_PROCESSED_DIR = BASE_DIR / "processed"


# ── 设备检测 ──────────────────────────────────────────────────────────────────
def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # RGT 的 TransformerConv 在 MPS 上有不支持的算子，默认回退 CPU
    return torch.device("cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="RGT OOD evaluation on Moltbook dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="已训练的 RGT checkpoint 路径（.pt）。\n"
             "若不指定，自动选取 checkpoints/ 目录下 F1 最高的文件。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        dest="checkpoint_dir",
        help=f"checkpoint 目录（默认：{DEFAULT_CHECKPOINT_DIR}）",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        dest="processed_dir",
        help=f"Moltbook RGT 预处理输出目录（默认：{DEFAULT_PROCESSED_DIR}）",
    )
    # 模型结构参数（须与训练时一致；默认值与 train.py 相同）
    parser.add_argument("--cat-num",          type=int,   default=2,   dest="cat_num")
    parser.add_argument("--numeric-num",      type=int,   default=5,   dest="numeric_num")
    parser.add_argument("--linear-channels",  type=int,   default=128, dest="linear_channels")
    parser.add_argument("--des-channel",      type=int,   default=768, dest="des_channel")
    parser.add_argument("--tweet-channel",    type=int,   default=768, dest="tweet_channel")
    parser.add_argument("--out-channel",      type=int,   default=128, dest="out_channel")
    parser.add_argument("--dropout",          type=float, default=0.5)
    parser.add_argument("--trans-head",       type=int,   default=8,   dest="trans_head")
    parser.add_argument("--semantic-head",    type=int,   default=8,   dest="semantic_head")
    parser.add_argument("--batch-size",       type=int,   default=256, dest="batch_size")
    return parser.parse_args()


# ── 自动选取最高 F1 checkpoint ────────────────────────────────────────────────
def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    从 ckpt_dir 中找文件名形如 best_checkpoint_f1{score}.pt 的文件，
    返回 F1 分数最高的那个。若目录为空或无匹配文件则报错。
    """
    candidates = list(ckpt_dir.glob("best_checkpoint_f1*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"在 {ckpt_dir} 中未找到 best_checkpoint_f1*.pt 文件。\n"
            "请先在 baselines-weibo/RGT/ 运行 train.py 生成检查点，\n"
            "或通过 --checkpoint 手动指定检查点路径。"
        )

    def parse_f1(p: Path) -> float:
        name = p.stem  # e.g. best_checkpoint_f10.6327
        # 文件名格式：best_checkpoint_f1{score}
        prefix = "best_checkpoint_f1"
        if name.startswith(prefix):
            try:
                return float(name[len(prefix):])
            except ValueError:
                pass
        return -1.0

    best = max(candidates, key=parse_f1)
    return best


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_data(processed_dir: Path):
    """
    加载预处理好的 Moltbook RGT 张量，构建 Data 对象。
    特征拼接顺序与 train.py 完全一致：
      x = [cat | num | tweet | des]
    """

    def load(name):
        return torch.load(processed_dir / name, map_location="cpu")

    cat_features   = load("cat_properties_tensor.pt")   # [N, 2]
    prop_features  = load("num_properties_tensor.pt")   # [N, 5]
    tweet_features = load("tweets_tensor.pt")            # [N, 768]
    des_features   = load("des_tensor.pt")               # [N, 768]
    edge_index     = load("edge_index.pt")               # [2, E]
    edge_type      = load("edge_type.pt").unsqueeze(-1)  # [E, 1]
    label          = load("label.pt")                    # [N]  全为 1

    # 拼接特征：cat | num | tweet | des（与 train.py 的切片顺序一致）
    x = torch.cat([cat_features, prop_features, tweet_features, des_features], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=label)

    # all nodes 都是 OOD agent 节点
    data.eval_idx = torch.arange(label.shape[0], dtype=torch.long)

    return data, cat_features.shape[1], prop_features.shape[1], \
           tweet_features.shape[1], des_features.shape[1]


# ── 模型（与 train.py 完全一致，直接复制） ────────────────────────────────────
class RGTDetector(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cat_num       = args.cat_num
        self.numeric_num   = args.numeric_num
        self.tweet_channel = args.tweet_channel
        self.des_channel   = args.des_channel

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

        self.drop = nn.Dropout(args.dropout)
        self.ReLU = nn.LeakyReLU()

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


# ── 推断 ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    """对 loader 内全部 mini-batch 做推断，返回 (preds_np, probs_np, labels_np)。"""
    model.eval()
    preds_list, probs_list, labels_list = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)
        # NeighborLoader 的 seed nodes 在 batch 的前 batch_size 个节点
        n_seed = batch.batch_size
        out_seed   = out[:n_seed]
        label_seed = batch.y[:n_seed]

        probs = torch.softmax(out_seed, dim=1)
        preds = torch.argmax(out_seed, dim=1)

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

    device = detect_device()
    print(f"Device: {device}")

    # ── 确定 checkpoint 路径 ─────────────────────────────────────────────────
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {ckpt_path}")
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_path = find_best_checkpoint(ckpt_dir)
    print(f"Checkpoint: {ckpt_path}")

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    processed_dir = Path(args.processed_dir)
    print(f"Loading data from {processed_dir} ...")
    data, cat_num, numeric_num, tweet_channel, des_channel = load_data(processed_dir)

    # 自动检测特征维度，若与默认参数不一致则覆盖并警告
    for attr, val, name in [
        ("cat_num",      cat_num,      "--cat-num"),
        ("numeric_num",  numeric_num,  "--numeric-num"),
        ("tweet_channel",tweet_channel,"--tweet-channel"),
        ("des_channel",  des_channel,  "--des-channel"),
    ]:
        if getattr(args, attr) != val:
            print(f"[Warning] {name}={getattr(args, attr)} but data has {val}; using {val}")
            setattr(args, attr, val)

    N = data.eval_idx.shape[0]
    print(f"OOD agent nodes: {N}")
    print(f"Edges          : {data.edge_index.shape[1]}")

    # ── 构建 DataLoader ───────────────────────────────────────────────────────
    eval_loader = NeighborLoader(
        data,
        input_nodes=data.eval_idx,
        num_neighbors=[20],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    model = RGTDetector(args).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # ── OOD 推断 ─────────────────────────────────────────────────────────────
    print(f"\n=== OOD Evaluation on Moltbook (all agents = bots, ground truth = 1, n={N}) ===")
    preds, probs, labels = evaluate(model, eval_loader, device)
    # 所有标签都已在预处理时设为 1，这里 labels 应全为 1
    print_metrics("[OOD]", preds, probs, labels)


if __name__ == "__main__":
    main()
