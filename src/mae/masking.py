from typing import Tuple
import torch


def make_token_mask(
    B: int,
    N: int,
    mask_ratio: float,
    mode: str,
    T: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns mask bool tensor: [B, N] where True means masked.
    mode:
      - "random": randomly mask tokens globally
      - "tube": mask the same spatial token positions across time (per-sample)
               (i.e., choose k spatial indices in [0..tokens_per_frame-1], apply to all frames)
    """
    k = max(1, int(N * mask_ratio))

    if mode == "random":
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:k]
            mask[b, idx] = True
        return mask

    if mode == "tube":
        # choose spatial indices, repeat across time
        k_spatial = max(1, int(tokens_per_frame * mask_ratio))
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        for b in range(B):
            sidx = torch.randperm(tokens_per_frame, device=device)[:k_spatial]  # [k_spatial]
            # expand to all frames
            # token index = t * tokens_per_frame + s
            tidx = (torch.arange(T, device=device).unsqueeze(1) * tokens_per_frame + sidx.unsqueeze(0)).reshape(-1)
            mask[b, tidx] = True
        return mask

    raise ValueError(f"Unknown mask mode: {mode}")
