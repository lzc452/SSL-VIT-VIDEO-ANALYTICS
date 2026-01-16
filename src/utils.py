import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"[ERROR] Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_dicts(path):
    """
    Return list of dict rows from a CSV file.
    """
    import csv
    path = Path(path)
    if not path.exists():
        print(f"[INFO] CSV missing, skip: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def write_csv(path, header, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def minmax_norm(values):
    """
    Normalize list of floats to [0,1]. If constant, return zeros.
    """
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return []
    mn = float(v.min())
    mx = float(v.max())
    if abs(mx - mn) < 1e-12:
        return [0.0 for _ in v]
    return ((v - mn) / (mx - mn)).tolist()

def save_checkpoint(state, filename):
    """保存模型权重到指定路径"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    # print(f"[INFO] Checkpoint saved to {filename}") # 可选：启用以确认保存
