"""
  python evaluate.py \\
      --model-path ../../baselines-weibo/BotDGT/output/Weibo/month+42+0.79354+0.28571.pt \\
      --data-dir ./data/Moltbook \\
      --interval month \\
      --window-size -1 \\
      --batch-size 64 \\
      --device auto \\
      --hidden-dim 128 \\
      --temporal-heads 4 \\
      --structural-heads 4 \\
      --temporal-drop 0.5 \\
      --structural-drop 0.0 \\
      --temporal-module attention
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import NeighborLoader

# ── 路径配置 & 依赖导入 ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
ROOT_DIR    = BASE_DIR.parent.parent
WEIBO_DGT   = ROOT_DIR / "baselines-weibo" / "BotDGT"

# 将 baselines-weibo/BotDGT 加入 sys.path，复用其 models/ 和 utils/ 代码
if str(WEIBO_DGT) not in sys.path:
    sys.path.insert(0, str(WEIBO_DGT))

from models.model import BotDyGNN                        # noqa: E402
from utils.metrics import compute_metrics_one_snapshot   # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    # 自动定位 Weibo BotDGT output 目录中唯一的 .pt 文件
    weibo_output_dir = WEIBO_DGT / "output" / "Weibo"
    default_ckpt = None
    if weibo_output_dir.exists():
        pt_files = sorted(weibo_output_dir.glob("*.pt"))
        if pt_files:
            default_ckpt = str(pt_files[-1])

    parser = argparse.ArgumentParser(
        description="BotDGT OOD evaluation on Moltbook dataset"
    )
    parser.add_argument(
        "--model-path", type=str, default=default_ckpt,
        help="训练好的 BotDyGNN checkpoint 路径（.pt state_dict）",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(BASE_DIR / "data" / "Moltbook"),
        help="Moltbook 预处理输出目录（默认：data/Moltbook）",
    )
    parser.add_argument(
        "--interval", type=str, default="month",
        choices=["year", "month", "three_months", "six_months",
                 "9_months", "15_months", "18_months", "21_months", "24_months"],
        help="时序快照采样间隔，须与训练时一致（默认：month）",
    )
    parser.add_argument(
        "--window-size", type=int, default=-1,
        help="时序窗口大小（-1 表示使用全部快照），须与训练时 checkpoint 的 window_size 一致（默认：-1）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="mini-batch 批大小（默认：64）",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="推断设备：auto / cuda / mps / cpu（默认：auto）",
    )
    # 模型超参（须与训练时一致，以正确初始化结构，再加载 state_dict）
    parser.add_argument("--hidden-dim",        type=int,   default=128)
    parser.add_argument("--temporal-heads",    type=int,   default=4)
    parser.add_argument("--structural-heads",  type=int,   default=4)
    parser.add_argument("--temporal-drop",     type=float, default=0.5)
    parser.add_argument("--structural-drop",   type=float, default=0.0)
    parser.add_argument(
        "--temporal-module", type=str, default="attention",
        choices=["attention", "gru", "lstm"],
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(BASE_DIR / "output"),
        help="结果保存目录（默认：output/）",
    )
    return parser.parse_args()


# ── 设备检测 ──────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"警告：当前环境不支持 CUDA，改用 CPU。")
            return "cpu"
        return requested
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("警告：当前环境不支持 MPS，改用 CPU。")
            return "cpu"
        return "mps"
    return "cpu"


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_graphs(data_dir: str, interval: str):
    """
    加载时序图快照，按 interval 从末尾向前采样，再反转使时间升序。
    与 baselines-weibo/BotDGT/utils/dataset.py 的 load_graphs() 逻辑完全一致。
    """
    interval_dict = {
        "year": 12, "month": 1, "three_months": 3, "six_months": 6,
        "9_months": 9, "15_months": 15, "18_months": 18,
        "21_months": 21, "24_months": 24,
    }
    step = interval_dict[interval]

    graph_dir = os.path.join(data_dir, "graph_data", "graphs")
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(
            f"图快照目录不存在：{graph_dir}\n"
            "请先运行预处理：python preprocess.py"
        )

    files = sorted(os.listdir(graph_dir))
    file_name = []
    for index in range(-1, -len(files) - 1, -step):
        file_name.append(files[index])
    file_name.reverse()
    print(f"加载 {len(file_name)} 个快照: {file_name[0]} → {file_name[-1]}")

    graph_list = [
        torch.load(os.path.join(graph_dir, f), weights_only=False)
        for f in file_name
    ]
    return graph_list


def load_processed_data(data_dir: str, device: str):
    """加载预处理后的特征张量和索引。"""
    base = os.path.join(data_dir, "processed_data")

    def pt(name):
        return torch.load(os.path.join(base, name), weights_only=True)

    des_tensor    = pt("des_tensor.pt").to(device)
    tweets_tensor = pt("tweets_tensor.pt").to(device)
    num_prop      = pt("num_properties_tensor.pt").to(device)
    cat_prop      = pt("cat_properties_tensor.pt").to(device)
    labels        = pt("label.pt").to(device)
    ood_idx       = pt("ood_idx.pt").to(device)

    return des_tensor, tweets_tensor, num_prop, cat_prop, labels, ood_idx


# ── 批次数据构建 ──────────────────────────────────────────────────────────────

class _DataLoader:
    """
    复用 baselines-weibo/BotDGT/utils/dataset.py 的 _DataLoader 逻辑。
    构建每个时序快照的 NeighborLoader。
    """
    def __init__(self, graphs, input_nodes, seed, batch_size):
        self.loader_list = [
            NeighborLoader(
                graph,
                shuffle=False,
                generator=torch.Generator().manual_seed(seed),
                batch_size=batch_size,
                input_nodes=input_nodes,
                num_neighbors=[-1, -1],
            )
            for graph in graphs
        ]
        self.iter_list = [iter(loader) for loader in self.loader_list]

    def iterate(self):
        return [next(it) for it in self.iter_list]


def build_batches(graphs, ood_idx, batch_size, seed, device):
    """
    为 OOD 用户集构建 mini-batch 数据，返回与 train.py 中 Dataset.get_final_data() 格式一致的数据结构。
    """
    graphs_on_cpu = [g.cpu() for g in graphs]
    ood_idx_cpu   = ood_idx.cpu()

    loader = _DataLoader(
        graphs=graphs_on_cpu,
        input_nodes=ood_idx_cpu,
        seed=seed,
        batch_size=batch_size,
    )

    total = len(ood_idx_cpu)
    all_right, all_n_id, all_edge_index = [], [], []
    all_edge_type, all_exist_nodes = [], []
    all_clustering_coefficient, all_bidirectional_links_ratio = [], []

    for i in range(0, total, batch_size):
        right = min(batch_size, total - i)
        all_right.append(right)
        subgraph_list = loader.iterate()
        all_n_id.append(              [sg.n_id.to("cpu")                        for sg in subgraph_list])
        all_edge_index.append(        [sg.edge_index.to("cpu")                  for sg in subgraph_list])
        all_edge_type.append(         [sg.edge_type.to("cpu")                   for sg in subgraph_list])
        all_exist_nodes.append(       [sg.exist_nodes.to("cpu")                 for sg in subgraph_list])
        all_clustering_coefficient.append(   [sg.clustering_coefficient.to("cpu")        for sg in subgraph_list])
        all_bidirectional_links_ratio.append([sg.bidirectional_links_ratio.to("cpu")     for sg in subgraph_list])

    return (
        all_right, all_n_id, all_edge_index, all_edge_type,
        all_exist_nodes, all_clustering_coefficient, all_bidirectional_links_ratio,
    )


# ── 模型初始化 & 加载 ─────────────────────────────────────────────────────────

class _FakeArgs:
    """用于初始化 BotDyGNN 的参数容器（取代 argparse.Namespace）。"""
    def __init__(self, hidden_dim, structural_head_config, structural_drop,
                 temporal_head_config, temporal_drop, window_size,
                 temporal_module_type):
        self.hidden_dim               = hidden_dim
        self.structural_head_config   = structural_head_config
        self.structural_drop          = structural_drop
        self.temporal_head_config     = temporal_head_config
        self.temporal_drop            = temporal_drop
        self.window_size              = window_size
        self.temporal_module_type     = temporal_module_type


def init_model(args, window_size: int, device: str):
    """
    用正确的 window_size（= 实际快照数量）初始化 BotDyGNN 模型结构。
    window_size 决定 GraphTemporalLayer 中 position_embedding_temporal 的维度。
    """
    fake_args = _FakeArgs(
        hidden_dim             = args.hidden_dim,
        structural_head_config = args.structural_heads,
        structural_drop        = args.structural_drop,
        temporal_head_config   = args.temporal_heads,
        temporal_drop          = args.temporal_drop,
        window_size            = window_size,
        temporal_module_type   = args.temporal_module,
    )
    model = BotDyGNN(fake_args)
    model.to(device)
    return model


# ── 推断 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ood_inference(model, args_device,
                      des_tensor, tweets_tensor, num_prop, cat_prop,
                      all_right, all_n_id, all_edge_index, all_edge_type,
                      all_exist_nodes, all_clustering_coefficient,
                      all_bidirectional_links_ratio):
    """
    对所有 OOD mini-batch 进行前向推断。
    ground truth 全为 1（机器人），取最后一个 snapshot 的预测结果。

    返回：
      all_preds  [N]   预测类别（0/1）
      all_probs  [N]   预测为机器人的概率
      all_labels [N]   真实标签（全 1）
    """
    model.eval()
    device = args_device
    all_preds, all_probs, all_labels = [], [], []

    for batch_size, bn_id, bei, bet, ben, bcc, bbi in zip(
        all_right, all_n_id, all_edge_index, all_edge_type,
        all_exist_nodes, all_clustering_coefficient, all_bidirectional_links_ratio
    ):
        des_list   = [des_tensor[n_id].to(device)    for n_id in bn_id]
        tweet_list = [tweets_tensor[n_id].to(device) for n_id in bn_id]
        num_list   = [num_prop[n_id].to(device)      for n_id in bn_id]
        cat_list   = [cat_prop[n_id].to(device)      for n_id in bn_id]
        edge_list  = [e.to(device)                   for e in bei]
        cc_list    = [c.to(device)                   for c in bcc]
        bi_list    = [b.to(device)                   for b in bbi]

        exist_nodes_tensor = torch.stack(
            [en[:batch_size].to(device) for en in ben], dim=0
        )  # [T, batch_size]

        # 前向（所有参数与 train.py 中的 forward_one_batch 一致）
        output = model(
            des_list, tweet_list, num_list, cat_list,
            edge_list, cc_list, bi_list,
            exist_nodes_tensor, batch_size,
        )
        # output: [batch_size, T, 2]，转置为 [T, batch_size, 2]
        output = output.transpose(0, 1)

        # 取最后一个 snapshot 的输出
        last_output = output[-1]          # [batch_size, 2]
        last_exist  = exist_nodes_tensor[-1]  # [batch_size]

        probs = F.softmax(last_output, dim=-1)[:, 1]  # 机器人概率
        preds = last_output.argmax(dim=-1)            # 预测类别

        # 只评估 exist_nodes=1 的节点（在该快照中已存在的用户）
        valid = last_exist == 1
        if valid.any():
            all_preds.extend(preds[valid].cpu().tolist())
            all_probs.extend(probs[valid].cpu().tolist())
            all_labels.extend([1] * int(valid.sum().item()))

    return all_preds, all_probs, all_labels


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_and_print_metrics(all_preds, all_probs, all_labels):
    n = len(all_preds)
    print(f"\n=== OOD 评估结果（共 {n} 个预测样本）===")

    if n == 0:
        print("警告：无有效预测样本（可能所有用户在最后一个快照都 exist_nodes=0）")
        return {}

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 若所有样本标签相同（全为 1），roc_auc_score 无法计算，做兼容处理
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    metrics = {
        "n_samples": n,
        "accuracy":  round(float(acc),  5),
        "precision": round(float(prec), 5),
        "recall":    round(float(rec),  5),
        "f1":        round(float(f1),   5),
        "auc":       round(float(auc),  5) if not __import__("math").isnan(auc) else "nan",
    }

    print(f"  Accuracy  : {metrics['accuracy']:.5f}")
    print(f"  Precision : {metrics['precision']:.5f}")
    print(f"  Recall    : {metrics['recall']:.5f}")
    print(f"  F1 (macro): {metrics['f1']:.5f}")
    print(f"  AUC       : {metrics['auc']}")
    print()

    # 预测标签分布
    bot_count   = sum(p == 1 for p in all_preds)
    human_count = sum(p == 0 for p in all_preds)
    print(f"  预测为机器人（1）: {bot_count}  预测为人类（0）: {human_count}")
    print(f"  （真实标签全为 1，Accuracy = bot 比例）")

    return metrics


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.model_path is None:
        raise ValueError(
            "--model-path 未指定，且在默认目录中未找到 .pt 文件。\n"
            "请先完成 Weibo BotDGT 训练，或手动指定 --model-path。"
        )

    device = resolve_device(args.device)
    print("=== BotDGT OOD Evaluation on Moltbook ===")
    print(f"Checkpoint : {args.model_path}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Device     : {device}")
    print(f"Interval   : {args.interval}")
    print(f"Window size: {args.window_size}")
    print()

    # ── 1. 加载图快照 ──────────────────────────────────────────────────────
    graphs = load_graphs(args.data_dir, args.interval)
    graphs = [g.to(device) for g in graphs]

    # 确定 actual_window_size（与训练时 Dataset 逻辑一致）
    if args.window_size == -1 or len(graphs) <= args.window_size:
        actual_window_size = len(graphs)
        print(f"使用全部 {actual_window_size} 个快照")
    else:
        actual_window_size = args.window_size
        graphs = graphs[-actual_window_size:]
        print(f"截取最近 {actual_window_size} 个快照（window_size={args.window_size}）")

    # ── 2. 加载特征数据 ────────────────────────────────────────────────────
    print("加载预处理特征...")
    des_tensor, tweets_tensor, num_prop, cat_prop, labels, ood_idx = \
        load_processed_data(args.data_dir, device)
    print(f"用户数 N = {des_tensor.shape[0]}, OOD 用户数 = {len(ood_idx)}")

    # ── 3. 初始化模型 ──────────────────────────────────────────────────────
    print(f"\n初始化 BotDyGNN (hidden_dim={args.hidden_dim}, window_size={actual_window_size})...")
    model = init_model(args, actual_window_size, device)

    # ── 4. 加载 checkpoint ────────────────────────────────────────────────
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Checkpoint 加载成功: {args.model_path}")

    # ── 5. 构建 OOD 批次 ──────────────────────────────────────────────────
    print(f"\n构建 OOD mini-batch（batch_size={args.batch_size}）...")
    (all_right, all_n_id, all_edge_index, all_edge_type,
     all_exist_nodes, all_clustering_coefficient,
     all_bidirectional_links_ratio) = build_batches(
        graphs, ood_idx, args.batch_size, seed=42, device=device,
    )

    # ── 6. 推断 ───────────────────────────────────────────────────────────
    print("开始 OOD 推断...")
    all_preds, all_probs, all_labels = run_ood_inference(
        model, device,
        des_tensor, tweets_tensor, num_prop, cat_prop,
        all_right, all_n_id, all_edge_index, all_edge_type,
        all_exist_nodes, all_clustering_coefficient, all_bidirectional_links_ratio,
    )

    # ── 7. 计算指标 ───────────────────────────────────────────────────────
    metrics = compute_and_print_metrics(all_preds, all_probs, all_labels)

    # ── 8. 保存结果 ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "ood_results.json"

    result = {
        "checkpoint": str(args.model_path),
        "data_dir":   str(args.data_dir),
        "interval":   args.interval,
        "window_size": actual_window_size,
        "n_snapshots": len(graphs),
        "metrics":    metrics,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {result_path}")


if __name__ == "__main__":
    main()
