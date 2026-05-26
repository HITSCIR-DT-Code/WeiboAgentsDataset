import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIBO_LABELS_PATH = PROJECT_ROOT / "dataset" / "Weibo" / "Weibo_Labels.json"
FOLLOWINGS_PATH = PROJECT_ROOT / "dataset" / "Weibo" / "followings.jsonl"
MOLTBOOK_POSTS_PATH = PROJECT_ROOT / "dataset" / "Moltbook" / "post_records.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "difference_analysis" / "plots" / "f-datasetstats-2.pdf"


# 三类账户颜色
LABEL_NAMES = {0: "Human", 2: "Bot", -1: "Agent"}
LABEL_COLORS = {0: "#30A46C", 2: "#E5484D", -1: "#4C9AFF"}
KEEP_LABELS = {0, 2, -1}

# 圆环图颜色
DONUT_COLORS = ["#4C9AFF", "#E5484D", "#30A46C", "#F5A524", "#8E4EC6"]


def load_weibo_subgraph():
    """加载微博三类账户的 followings 子图。"""
    with open(WEIBO_LABELS_PATH, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)

    # 仅保留 label ∈ {0, 2, -1}
    uid2label = {uid: lab for uid, lab in raw_labels.items() if lab in KEEP_LABELS}
    keep_uids = set(uid2label.keys())

    edges = []
    with open(FOLLOWINGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s, t = d.get("source_uid"), d.get("target_uid")
            if s in keep_uids and t in keep_uids and s != t:
                edges.append((s, t))

    G = nx.Graph()
    G.add_nodes_from(keep_uids)
    G.add_edges_from(edges)

    # 仅保留度数 >= 1 的节点
    isolated = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolated)

    return G, uid2label


def draw_network(ax, G, uid2label):
    """在 ax 上绘制网络图。"""
    # 节点按类别分组
    nodes_by_label = {0: [], 2: [], -1: []}
    for n in G.nodes():
        lab = uid2label.get(n)
        if lab in nodes_by_label:
            nodes_by_label[lab].append(n)

    print("[Weibo network]")
    print(f"  nodes (degree>=1): {G.number_of_nodes()}")
    print(f"  edges            : {G.number_of_edges()}")
    for lab, ns in nodes_by_label.items():
        print(f"  {LABEL_NAMES[lab]:6s} (label={lab}): {len(ns)}")

    print("  computing spring_layout ...")
    # k 设为远小于默认 1/sqrt(n) 的值让节点更聚拢；iterations 增大到充分收敛
    n_nodes = G.number_of_nodes()
    k_val = 1 / (n_nodes ** 0.5)
    pos = nx.spring_layout(
        G, k=k_val, iterations=150, threshold=1e-5, seed=42
    )

    # 边
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#CCCCCC", alpha=0.3, width=0.3
    )

    # 按类别绘制节点（便于 legend）
    for lab in [0, 2, -1]:
        ns = nodes_by_label[lab]
        if not ns:
            continue
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ns,
            node_color=LABEL_COLORS[lab],
            node_size=20,
            ax=ax,
            label=f"{LABEL_NAMES[lab]}",
            linewidths=0,
        )

    import numpy as np

    # 收紧视图范围，让网络图更贴近边界
    all_pos = np.array(list(pos.values()))
    min_xy = all_pos.min(axis=0)
    max_xy = all_pos.max(axis=0)
    margin = (max_xy - min_xy) * 0.01  # 仅留 1% 边距
    ax.set_xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
    ax.set_ylim(min_xy[1] - margin[1], max_xy[1] + margin[1])

    ax.set_title("(a) Weibo Account Network", fontsize=30, pad=4)
    ax.legend(loc="upper left", fontsize=26, frameon=True,
              handletextpad=0.4, handlelength=0.8, labelspacing=0.4)
    ax.axis("off")


def draw_donut(ax):
    """在 ax 上绘制 Moltbook 领域圆环图。"""
    import numpy as np

    counter = Counter()
    with open(MOLTBOOK_POSTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sm = d.get("submolt")
            if sm is not None:
                counter[sm] += 1

    items = counter.most_common()
    labels = [k for k, _ in items]
    sizes = [v for _, v in items]
    total = sum(sizes)
    colors = DONUT_COLORS[: len(items)]

    print("[Moltbook submolt]")
    for k, v in items:
        print(f"  {k:15s}: {v}")
    print(f"  total          : {total}")

    def _autopct(p):
        return f"{p:.1f}%"

    wedges, _texts, _autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=_autopct,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        pctdistance=0.8,
        textprops=dict(fontsize=24),
    )

    # 为每段圆环添加浮动标签
    for i, (wedge, (name, cnt)) in enumerate(zip(wedges, items)):
        # 计算扇区中间角度
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        ang_rad = np.deg2rad(ang)

        # 标签位置（圆环外侧，线短一点）
        x = 1.12 * np.cos(ang_rad)
        y = 1.12 * np.sin(ang_rad)

        # 连接线拐点
        conn_x = 1.02 * np.cos(ang_rad)
        conn_y = 1.02 * np.sin(ang_rad)

        label_text = f"{name.upper()}"

        ha = "left" if x >= 0 else "right"

        ax.annotate(
            label_text,
            xy=(conn_x, conn_y),
            xytext=(x, y),
            fontsize=20,
            ha=ha,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=colors[i],
                lw=1.5,
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors[i],
                alpha=0.15,
                edgecolor=colors[i],
                linewidth=1.2,
            ),
        )

    ax.set_title("(b) Post Distribution by Submolt", fontsize=30, pad=4)
    ax.set(aspect="equal")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

    G, uid2label = load_weibo_subgraph()
    draw_network(ax_left, G, uid2label)

    draw_donut(ax_right)

    #画图
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSaved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
