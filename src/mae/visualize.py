import os
from typing import Optional
import torch
import torchvision.utils as vutils


@torch.no_grad()
def save_debug_grid(
    clip: torch.Tensor,
    save_path: str,
    max_frames: int = 8,
) -> None:
    """
    简单把输入 clip 的前几帧保存出来，确认数据管道没问题。
    clip: [B,C,T,H,W]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    B, C, T, H, W = clip.shape
    t = min(T, max_frames)
    # take first sample
    x = clip[0, :, :t]  # [C,t,H,W]
    x = x.permute(1, 0, 2, 3)  # [t,C,H,W]
    # unnormalize is skipped (just debug)
    grid = vutils.make_grid(x, nrow=t, normalize=True, scale_each=True)
    vutils.save_image(grid, save_path)
