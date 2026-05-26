import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from model import BotRGCN
from utils import accuracy, init_weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "weibo-preprocess" / "weibo_processed_data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# ---------------------------------------------------------------------------
# Device detection: CUDA > MPS > CPU
# ---------------------------------------------------------------------------
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = detect_device()
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description='BotRGCN on Weibo')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='启用后使用训练集类别频率的倒数对 CrossEntropyLoss 加权，以缓解类别不平衡')
    return parser.parse_args()

args = _parse_args()

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
embedding_size = 128
dropout        = 0.2
lr             = 1e-3
weight_decay   = 1e-4
epochs         = 250

# ---------------------------------------------------------------------------
# Load preprocessed tensors
# ---------------------------------------------------------------------------
def load(filename):
    return torch.load(DATA_DIR / filename, map_location=device)

des_tensor       = load("des_tensor.pt")
tweets_tensor    = load("tweets_tensor.pt")
num_prop         = load("num_properties_tensor.pt")
cat_prop         = load("cat_properties_tensor.pt")
edge_index       = load("edge_index.pt")
edge_type        = load("edge_type.pt")
labels           = load("label.pt")        # 0=human, 1=bot, 2=agent(OOD)
train_idx        = load("train_idx.pt")
val_idx          = load("val_idx.pt")
test_idx         = load("test_idx.pt")

# Indices of agent nodes (OOD), not in any split
agent_idx = (labels == 2).nonzero(as_tuple=True)[0]


def ensure_bidirectional_relations(edge_index_tensor, edge_type_tensor):
    unique_types = torch.unique(edge_type_tensor)
    if unique_types.numel() > 1:
        return edge_index_tensor, edge_type_tensor

    reverse_edge_index = torch.stack((edge_index_tensor[1], edge_index_tensor[0]), dim=0)
    reverse_edge_type = torch.ones(reverse_edge_index.shape[1], dtype=edge_type_tensor.dtype, device=edge_type_tensor.device)

    augmented_edge_index = torch.cat((edge_index_tensor, reverse_edge_index), dim=1)
    augmented_edge_type = torch.cat((edge_type_tensor, reverse_edge_type), dim=0)
    return augmented_edge_index, augmented_edge_type


edge_index, edge_type = ensure_bidirectional_relations(edge_index, edge_type)


def build_class_weight(train_labels):
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = class_counts.sum() / (class_counts.clamp_min(1.0) * class_counts.numel())
    return class_weights.to(device)


# ---------------------------------------------------------------------------
# Model, loss, optimiser
# ---------------------------------------------------------------------------
model = BotRGCN(
    num_prop_size=num_prop.shape[1],   # 5
    cat_prop_size=cat_prop.shape[1],   # 1
    embedding_dimension=embedding_size,
    dropout=dropout,
).to(device)

if args.weighted_loss:
    class_weight = build_class_weight(labels[train_idx])
    print(f"[weighted_loss] class weights: human={class_weight[0]:.4f}, bot={class_weight[1]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

model.apply(init_weights)


def collect_metrics(logits, target):
    probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
    predictions = logits.argmax(dim=1).detach().cpu()
    target_cpu = target.detach().cpu()

    metrics = {
        "accuracy": (predictions == target_cpu).float().mean().item(),
        "precision": precision_score(target_cpu.numpy(), predictions.numpy(), zero_division=0, average="macro"),
        "recall": recall_score(target_cpu.numpy(), predictions.numpy(), zero_division=0, average="macro"),
        "f1": f1_score(target_cpu.numpy(), predictions.numpy(), zero_division=0, average="macro"),
    }

    unique_target = torch.unique(target_cpu)
    if unique_target.numel() > 1:
        metrics["auc"] = roc_auc_score(target_cpu.numpy(), probabilities.numpy())
    else:
        metrics["auc"] = float("nan")
    return metrics

# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------
def train(epoch):
    model.train()
    output = model(des_tensor, tweets_tensor, num_prop, cat_prop, edge_index, edge_type)
    loss_train = criterion(output[train_idx], labels[train_idx])
    acc_train  = accuracy(output[train_idx], labels[train_idx])
    optimizer.zero_grad()
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_output = model(des_tensor, tweets_tensor, num_prop, cat_prop, edge_index, edge_type)
        val_metrics = collect_metrics(eval_output[val_idx], labels[val_idx])

    print(
        f"Epoch: {epoch+1:04d}  "
        f"loss_train: {loss_train.item():.4f}  "
        f"acc_train: {acc_train.item():.4f}  "
        f"acc_val: {val_metrics['accuracy']:.4f}  "
        f"f1_val: {val_metrics['f1']:.4f}"
    )
    return {
        "loss": loss_train.item(),
        "train_acc": acc_train.item(),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
    }


# ---------------------------------------------------------------------------
# Final test + OOD evaluation
# ---------------------------------------------------------------------------
def test():
    model.eval()
    with torch.no_grad():
        output = model(des_tensor, tweets_tensor, num_prop, cat_prop, edge_index, edge_type)

    # ---- In-distribution test set ----
    loss_test = criterion(output[test_idx], labels[test_idx])
    test_metrics = collect_metrics(output[test_idx], labels[test_idx])

    print("\n=== In-distribution Test Results ===")
    print(
        f"  accuracy:  {test_metrics['accuracy']:.4f}\n"
        f"  precision: {test_metrics['precision']:.4f}\n"
        f"  recall:    {test_metrics['recall']:.4f}\n"
        f"  f1:        {test_metrics['f1']:.4f}\n"
        f"  auc:       {test_metrics['auc']:.4f}"
    )

    # ---- OOD evaluation (agent nodes, ground truth = 1 / bot) ----
    if len(agent_idx) == 0:
        print("\n[OOD] No agent nodes found, skipping OOD evaluation.")
        return

    ood_target = torch.ones(len(agent_idx), dtype=labels.dtype, device=device)
    ood_loss = criterion(output[agent_idx], ood_target)
    ood_metrics = collect_metrics(output[agent_idx], ood_target)

    print("\n=== OOD Evaluation (Agent nodes, treated as bot-like positives) ===")
    print(
        f"  num_agents: {len(agent_idx)}\n"
        f"  accuracy:  {ood_metrics['accuracy']:.4f}\n"
        f"  precision: {ood_metrics['precision']:.4f}\n"
        f"  recall:    {ood_metrics['recall']:.4f}\n"
        f"  f1:        {ood_metrics['f1']:.4f}\n"
        f"  auc:       {ood_metrics['auc']:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_checkpoint_path = CHECKPOINT_DIR / f"best_model_{run_id}.pt"

    for epoch in range(epochs):
        metrics = train(epoch)
        if metrics["val_f1"] >= best_val_f1:
            best_val_f1 = metrics["val_f1"]
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "weighted_loss": args.weighted_loss,
                },
                best_checkpoint_path,
            )
            print(f"Saved best checkpoint to {best_checkpoint_path} (epoch={best_epoch}, val_f1={best_val_f1:.4f})")

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"\nLoaded best checkpoint from {best_checkpoint_path} "
            f"(epoch={checkpoint['epoch']}, val_f1={checkpoint['val_f1']:.4f})"
        )
    elif best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best checkpoint from memory with validation f1={best_val_f1:.4f}")

    test()
