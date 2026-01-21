# src/mae/metrics.py
from __future__ import annotations

from typing import Dict, Any


def update_best(best: Dict[str, Any], stats: Dict[str, float], key: str = "loss", mode: str = "min") -> Dict[str, Any]:
    cur = float(stats.get(key, 1e9))
    if mode == "min":
        if cur < float(best.get(key, 1e9)):
            return {**best, key: cur, "epoch": int(stats.get("epoch", best.get("epoch", 0)))}
        return best
    else:
        if cur > float(best.get(key, -1e9)):
            return {**best, key: cur, "epoch": int(stats.get("epoch", best.get("epoch", 0)))}
        return best


def format_metrics(epoch: int, stats: Dict[str, float], best: Dict[str, Any]) -> str:
    return (
        f"ep={epoch} "
        f"loss={stats['loss']:.4f} l1={stats['l1']:.4f} l2={stats['l2']:.4f} "
        f"std_pred={stats['pred_std']:.3f} std_tgt={stats['target_std']:.3f} "
        f"mask={stats['mask_ratio']:.2f} "
        f"time={stats['time_sec']:.1f}s "
        f"best_loss={best.get('loss', 0):.4f}@{best.get('epoch', 0)}"
    )
