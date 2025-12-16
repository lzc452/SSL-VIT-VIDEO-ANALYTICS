from collections import defaultdict
from pathlib import Path
import random


def read_split(split_file):
    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, y = line.split()
            items.append((p, int(y)))
    return items


def write_split(items, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p, y in items:
            f.write(f"{p} {y}\n")


def make_class_shard_splits(
    base_split_file,
    num_clients,
    shards_per_client=6,
    seed=42,
    min_samples_per_client=200,
    out_prefix="fed",
    out_dir="data/splits",
):
    """
    Class-shard Non-IID:
    - group samples by label
    - create shards from labels, then assign shards to clients

    Output:
      data/splits/{out_prefix}_client_{i}_train.txt
    """
    rng = random.Random(seed)

    items = read_split(base_split_file)
    by_class = defaultdict(list)
    for p, y in items:
        by_class[y].append((p, y))

    # shuffle within each class
    for y in by_class:
        rng.shuffle(by_class[y])

    # create shards: each shard is a chunk from a class
    # strategy: split each class list into 1 shard (simple) then assign classes to clients in shards_per_client groups
    # to increase non-IID strength, we allocate class lists as "shards"
    class_ids = sorted(by_class.keys())
    rng.shuffle(class_ids)

    # Assign shards_per_client classes to each client (with wrap-around)
    client_items = [[] for _ in range(num_clients)]
    idx = 0
    for cid in class_ids:
        client_id = (idx // shards_per_client) % num_clients
        client_items[client_id].extend(by_class[cid])
        idx += 1

    # check min samples; if too small, rebalance by moving from largest to smallest
    def sizes():
        return [len(ci) for ci in client_items]

    # simple rebalance loop
    for _ in range(200):
        s = sizes()
        mn = min(s)
        mx = max(s)
        if mn >= min_samples_per_client:
            break
        small = s.index(mn)
        large = s.index(mx)
        if len(client_items[large]) <= min_samples_per_client:
            break
        # move some samples
        move_n = min(200, len(client_items[large]) - min_samples_per_client)
        client_items[small].extend(client_items[large][:move_n])
        client_items[large] = client_items[large][move_n:]

    out_paths = []
    out_stats = []
    out_dir = Path(out_dir)

    for i in range(num_clients):
        out_path = out_dir / f"{out_prefix}_client_{i}_train.txt"
        write_split(client_items[i], out_path)
        out_paths.append(str(out_path))

        cls_set = sorted({y for _, y in client_items[i]})
        out_stats.append({
            "client": i,
            "num_samples": len(client_items[i]),
            "num_classes": len(cls_set),
            "classes": " ".join(map(str, cls_set[:50]))  # avoid huge string
        })

    return out_paths, out_stats
