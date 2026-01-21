# src/mae/losses.py
"""
Loss functions for MAE-style video SSL.

Design principles:
- No numpy
- No exotic CUDA ops
- Stable on Hopper (H200)
- Paper-friendly decomposition
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Reconstruction losses
# ------------------------------------------------------------
def mae_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Standard MAE L2 reconstruction loss.

    Args:
        pred:   [N_mask, D]
        target: [N_mask, D]
        normalize: if True, normalize target per-token (MAE default)

    Returns:
        scalar loss
    """
    if normalize:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / (var + 1.0e-6).sqrt()

    loss = (pred - target) ** 2
    return loss.mean()


def mae_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """L1 reconstruction loss (optional ablation)."""
    return torch.abs(pred - target).mean()


def mae_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cosine similarity loss.
    Useful when magnitude is less important than direction.
    """
    pred = F.normalize(pred, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    return 1.0 - (pred * target).sum(dim=-1).mean()


# ------------------------------------------------------------
# Combined loss entry
# ------------------------------------------------------------
def build_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l2",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Unified MAE loss entry.

    loss_type:
        - "l2"      : standard MAE (default, recommended)
        - "l1"      : ablation
        - "cosine"  : direction-only ablation
    """
    if loss_type == "l2":
        return mae_l2_loss(pred, target, normalize=normalize)
    elif loss_type == "l1":
        return mae_l1_loss(pred, target)
    elif loss_type == "cosine":
        return mae_cosine_loss(pred, target)
    else:
        raise ValueError(f"Unknown MAE loss type: {loss_type}")


# ------------------------------------------------------------
# Diagnostics (optional but useful)
# ------------------------------------------------------------
@torch.no_grad()
def reconstruction_error_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Returns simple statistics for logging / debugging."""
    err = pred - target
    return {
        "mae_l1": torch.abs(err).mean().item(),
        "mae_l2": (err ** 2).mean().item(),
        "pred_std": pred.std(dim=0).mean().item(),
        "target_std": target.std(dim=0).mean().item(),
    }

# ------------------------------------------------------------
# Backward-compatible alias (do NOT remove)
# engine.py (old versions) may import `mae_recon_loss`.
# We keep this wrapper to avoid breaking training scripts.
# ------------------------------------------------------------
def mae_recon_loss(pred, target, *args, **kwargs):
    """
    Backward-compatible wrapper for older engine.py.

    Supported call patterns:
      - mae_recon_loss(pred, target)
      - mae_recon_loss(pred, target, normalize_target_bool)
      - mae_recon_loss(pred, target, loss_type_str, normalize_target_bool)
      - mae_recon_loss(pred, target, loss_type="l2", normalize_target=True)
      - mae_recon_loss(pred, target, normalize=True)  # alias
    """
    # defaults aligned with build_mae_loss
    loss_type = kwargs.pop("loss_type", "l2")

    # normalize flag: accept both names
    if "normalize_target" in kwargs:
        normalize = bool(kwargs.pop("normalize_target"))
    else:
        normalize = bool(kwargs.pop("normalize", True))

    # positional fallback (keep very tolerant)
    if len(args) == 1:
        # treat as normalize_target
        normalize = bool(args[0])
    elif len(args) >= 2:
        # (loss_type, normalize_target)
        loss_type = str(args[0])
        normalize = bool(args[1])

    return build_mae_loss(pred, target, loss_type=loss_type, normalize=normalize)