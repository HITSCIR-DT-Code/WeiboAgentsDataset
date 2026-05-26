import json
import random
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

with open("Weibo_Labels.json", "r") as f:
    labels = json.load(f)

groups = defaultdict(list)
ood_ids = []

for uid, label in labels.items():
    if label == 1:
        continue  # 丢弃疑似类别
    elif label == -1:
        ood_ids.append(uid)
    else:
        groups[label].append(uid)

# 分层划分 0 和 2
train_ids, val_ids, test_ids = [], [], []

rng = random.Random(SEED)

for label, uids in groups.items():
    uids = sorted(uids)  # 保证可复现
    rng.shuffle(uids)
    n = len(uids)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    train_ids.extend(uids[:n_train])
    val_ids.extend(uids[n_train:n_train + n_val])
    test_ids.extend(uids[n_train + n_val:])

# 打印统计
for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    cnt = defaultdict(int)
    for uid in split_ids:
        cnt[labels[uid]] += 1
    print(f"{split_name}: total={len(split_ids)}, {dict(cnt)}")

print(f"ood (-1): {len(ood_ids)}")

# 保存结果
split = {}
for uid in train_ids:
    split[uid] = "train"
for uid in val_ids:
    split[uid] = "val"
for uid in test_ids:
    split[uid] = "test"
for uid in ood_ids:
    split[uid] = "ood"

with open("split.json", "w") as f:
    json.dump(split, f, indent=2, ensure_ascii=False)

print("Saved: split.json")
