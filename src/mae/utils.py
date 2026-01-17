import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class Logger:
    log_file: str

    def write(self, msg: str) -> None:
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def dump_config(cfg: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def save_checkpoint(
    save_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    obj = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, save_path)


def keep_last_n_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    p = Path(ckpt_dir)
    if not p.exists():
        return
    ckpts = sorted([x for x in p.glob("*.pth")], key=lambda x: x.stat().st_mtime)
    if len(ckpts) <= keep_last:
        return
    for x in ckpts[:-keep_last]:
        try:
            x.unlink()
        except Exception:
            pass
