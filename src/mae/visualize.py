# src/mae/visualize.py
from __future__ import annotations

import os
from typing import Any, Dict

import torch


def maybe_save_vis(
    epoch: int,
    model,
    dataset,
    device: torch.device,
    cfg: Dict[str, Any],
    out_dir: str,
    logger,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        logger.write("[VIS] matplotlib not available -> skip")
        return

    model.eval()
    model_cfg = cfg.get("model", {})
    mae_cfg = cfg.get("mae", {})

    stage4_pool = int(model_cfg.get("stage4_pool", 3))
    P = stage4_pool * stage4_pool

    mask_ratio = float(mae_cfg.get("mask_ratio", 0.8))
    mask_mode = str(mae_cfg.get("mask_mode", "tube"))

    from .masking import make_token_mask

    clip = dataset[0]
    if clip.dim() == 4:
        clip = clip.unsqueeze(0)
    clip = clip.to(device)
    B = clip.size(0)
    T = clip.size(2)

    mask = make_token_mask(B, T, P, mask_ratio, mask_mode, device=device)

    with torch.no_grad():
        pred, tgt = model(clip, token_mask=mask, stage4_pool=stage4_pool)
        err = (pred - tgt).pow(2).mean(dim=-1)  # [B, M]
        err = err[0].detach().cpu().float().numpy()

    mask_np = mask[0].detach().cpu().numpy().astype("float32")  # [N]
    mask_grid = mask_np.reshape(T, P)

    # reconstruct full grid error: masked positions get err, visible -> 0
    err_full = (mask_np * 0.0).reshape(-1)
    err_full[mask_np.astype(bool)] = err
    err_grid = err_full.reshape(T, P)

    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # save mask grid
    plt.figure()
    plt.imshow(mask_grid, aspect="auto")
    plt.title(f"mask grid ep{epoch:03d} mode={mask_mode} ratio={mask_ratio:.2f}")
    plt.xlabel("spatial token")
    plt.ylabel("time")
    mask_path = os.path.join(vis_dir, f"mask_ep{epoch:03d}.png")
    plt.savefig(mask_path, dpi=150, bbox_inches="tight")
    plt.close()

    # save error heatmap
    plt.figure()
    plt.imshow(err_grid, aspect="auto")
    plt.title(f"masked token error ep{epoch:03d} (visible=0)")
    plt.xlabel("spatial token")
    plt.ylabel("time")
    err_path = os.path.join(vis_dir, f"err_ep{epoch:03d}.png")
    plt.savefig(err_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.write(f"[VIS] saved: {os.path.basename(mask_path)}, {os.path.basename(err_path)}")
    model.train()
