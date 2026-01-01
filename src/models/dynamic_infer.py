"""src/models/dynamic_infer.py

Journal-grade dynamic inference utilities (video-level) for edge deployment.

Key idea
--------
Most "early-exit" demo code re-runs the backbone on the same prefix multiple times
to check the confidence at different prefix lengths. That *does not* represent
real compute savings on edge devices.

Here we implement *streaming* inference:
  - frames are encoded *once* in temporal order;
  - a running mean embedding is maintained;
  - confidence is checked after each newly processed frame;
  - samples that satisfy the confidence threshold stop consuming compute.

We also include a lightweight frame gating method based on frame-difference
motion scores, and a hybrid mode (gating + early-exit).

All functions are designed to integrate with the repo's MobileViT-S backbone
and the finetune classifier head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def motion_scores_l1(clip: torch.Tensor) -> torch.Tensor:
    """Compute per-frame motion score with cheap L1 differences.

    Args:
        clip: [B, C, T, H, W] normalized tensor.

    Returns:
        scores: [B, T], higher means more motion/change.
    """
    B, C, T, H, W = clip.shape
    scores = torch.zeros((B, T), device=clip.device)
    if T <= 1:
        return scores
    diffs = (clip[:, :, 1:, :, :] - clip[:, :, :-1, :, :]).abs().mean(dim=(1, 3, 4))  # [B, T-1]
    scores[:, 1:] = diffs
    return scores


@torch.no_grad()
def select_topk_frames(clip: torch.Tensor, k: int, score_type: str = "motion") -> Tuple[torch.Tensor, torch.Tensor]:
    """Select k frames per sample and return the selected clip.

    Args:
        clip: [B,C,T,H,W]
        k: number of frames to keep
        score_type: "motion" | "random"

    Returns:
        clip_sel: [B,C,k,H,W]
        idx: [B,k] selected indices (sorted ascending)
    """
    B, C, T, H, W = clip.shape
    k_eff = min(int(k), T)

    if score_type == "motion":
        scores = motion_scores_l1(clip)  # [B,T]
        idx = scores.topk(k_eff, dim=1, largest=True).indices
        idx, _ = idx.sort(dim=1)
    elif score_type == "random":
        idx = torch.stack([
            torch.randperm(T, device=clip.device)[:k_eff].sort()[0]
            for _ in range(B)
        ], dim=0)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    idx_view = idx.view(B, 1, k_eff, 1, 1).expand(B, C, k_eff, H, W)
    clip_sel = torch.gather(clip, dim=2, index=idx_view)
    return clip_sel, idx


@dataclass
class EarlyExitStats:
    """Statistics returned by streaming early-exit."""
    used_frames: torch.Tensor  # [B] int
    final_conf: torch.Tensor   # [B] float


@torch.no_grad()
def streaming_early_exit(
    backbone,
    classifier,
    clip: torch.Tensor,
    threshold: float,
    min_frames: int = 4,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
) -> Tuple[torch.Tensor, EarlyExitStats]:
    """Streaming confidence-based temporal early-exit.

    This implementation processes each frame at most once.

    Args:
        backbone: MobileViT-S backbone. Must support `backbone(x) -> (feat_map, emb)` or similar.
        classifier: nn.Linear mapping embedding -> logits.
        clip: [B,C,T,H,W]
        threshold: confidence threshold (max softmax prob)
        min_frames: minimum frames before allowing exit
        max_frames: optional cap on frames to process
        frame_step: process every `frame_step` frames (>=1). Extra compute knob.

    Returns:
        final_logits: [B,K]
        stats: EarlyExitStats (used_frames, final_conf)
    """
    device = clip.device
    B, C, T, H, W = clip.shape
    if max_frames is not None:
        T = min(T, int(max_frames))
        clip = clip[:, :, :T, :, :]
    frame_step = max(int(frame_step), 1)
    min_frames = max(int(min_frames), 1)

    used = torch.zeros((B,), dtype=torch.long, device=device)
    decided = torch.zeros((B,), dtype=torch.bool, device=device)

    sum_emb = None
    cnt_emb = torch.zeros((B,), dtype=torch.long, device=device)
    final_logits = None

    def _encode_frame(x: torch.Tensor) -> torch.Tensor:
        out = backbone(x)
        # repo MobileViT returns (feat_map, emb)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            emb = out[1]
        else:
            feat = out
            emb = feat.mean(dim=[2, 3])
        return emb

    emb0 = _encode_frame(clip[:, :, 0, :, :])  # [B,D]
    D = emb0.shape[-1]
    sum_emb = torch.zeros((B, D), device=device, dtype=emb0.dtype)
    final_logits = torch.zeros((B, classifier.out_features), device=device, dtype=emb0.dtype)

    sum_emb += emb0
    cnt_emb += 1

    def _check_and_update(active_mask: torch.Tensor):
        if not active_mask.any():
            return
        mean_emb = sum_emb[active_mask] / cnt_emb[active_mask].unsqueeze(1).clamp_min(1)
        logits = classifier(mean_emb)
        prob = F.softmax(logits, dim=1)
        conf, _ = prob.max(dim=1)

        can_exit = cnt_emb[active_mask] >= min_frames
        newly = (conf >= threshold) & can_exit
        if newly.any():
            idx = active_mask.nonzero(as_tuple=False).view(-1)
            idx_new = idx[newly]
            final_logits[idx_new] = logits[newly]
            used[idx_new] = cnt_emb[idx_new]
            decided[idx_new] = True

    _check_and_update(~decided)

    for t in range(1, T, frame_step):
        active = ~decided
        if not active.any():
            break
        emb = _encode_frame(clip[:, :, t, :, :])
        sum_emb[active] += emb[active]
        cnt_emb[active] += 1
        _check_and_update(active)

    remain = ~decided
    if remain.any():
        mean_emb = sum_emb[remain] / cnt_emb[remain].unsqueeze(1).clamp_min(1)
        logits = classifier(mean_emb)
        final_logits[remain] = logits
        used[remain] = cnt_emb[remain]
        decided[remain] = True

    final_conf = F.softmax(final_logits, dim=1).max(dim=1)[0]
    return final_logits, EarlyExitStats(used_frames=used, final_conf=final_conf)
