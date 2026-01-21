# src/mae/masking.py
from __future__ import annotations

from typing import Any, Dict, List

import torch


def get_mask_ratio(epoch: int, schedule: List[Dict[str, Any]], default: float) -> float:
    if not schedule:
        return float(default)
    for seg in schedule:
        s = int(seg.get("start", 1))
        e = int(seg.get("end", 10**9))
        v = float(seg.get("value", default))
        if s <= epoch < e:
            return v
    return float(default)


def make_token_mask(
    B: int,
    T: int,
    tokens_per_frame: int,
    mask_ratio: float,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Return:
      mask: [B, N] bool, True means masked
      N = T * tokens_per_frame
    """
    P = tokens_per_frame
    N = T * P
    num_mask = max(1, int(round(N * mask_ratio)))

    if mode == "random":
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:num_mask]
            mask[b, idx] = True
        return mask

    if mode == "tube":
        # tube: choose spatial tokens, apply across all frames
        # per-frame masked count = round(P * mask_ratio)
        m_pf = max(1, int(round(P * mask_ratio)))
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        for b in range(B):
            sidx = torch.randperm(P, device=device)[:m_pf]  # spatial indices
            # broadcast across time
            for t in range(T):
                mask[b, t * P + sidx] = True
        return mask

    raise ValueError(f"Unknown mask_mode: {mode}")


def count_masked(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def count_visible(mask: torch.Tensor) -> int:
    return int((~mask).sum().item())
