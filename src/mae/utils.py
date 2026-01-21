# src/mae/utils.py
from __future__ import annotations

import os
import json
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # speed-friendly deterministic policy (paper-acceptable)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    """
    Make DataLoader workers deterministic with the same global seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class Logger:
    path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, msg: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "best": best,
        "cfg_hash": config_hash(cfg),
    }
    torch.save(ckpt, path)


def keep_last_n_checkpoints(out_dir: str, prefix: str, keep: int, suffix: str = ".pth") -> None:
    files = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith(suffix)]
    if len(files) <= keep:
        return
    files.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)))
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
