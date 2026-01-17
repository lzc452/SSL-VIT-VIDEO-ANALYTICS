import math
import time
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .masking import make_token_mask
from .losses import total_loss
from .metrics import diag_stats


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    scfg = cfg.get("training", {}).get("scheduler", {})
    if not scfg or not scfg.get("enable", True):
        return None

    warmup_epochs = int(scfg.get("warmup_epochs", 10))
    eta_min_ratio = float(scfg.get("eta_min_ratio", 0.05))
    epochs = int(cfg["training"]["epochs"])

    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)

    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * eta_min_ratio

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return (eta_min / base_lr) + 0.5 * (1.0 - (eta_min / base_lr)) * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    cfg: Dict[str, Any],
    logger,
):
    model.train()

    tr = cfg["training"]
    mae_cfg = cfg["mae"]
    mcfg = cfg["model"]

    amp_enabled = bool(tr.get("amp", False) and device.type == "cuda")
    grad_accum = int(tr.get("grad_accum", 1))
    clip_norm = float(tr.get("clip_grad_norm", 1.0))
    log_interval = int(tr.get("log_interval", 20))

    mask_ratio = float(mae_cfg.get("mask_ratio", 0.9))
    mask_mode = str(mae_cfg.get("mask_mode", "tube"))

    tokens_per_frame = int((mcfg.get("stage4_pool", 3)) ** 2)
    T = int(cfg["dataset"]["clip_len"])

    t0 = time.time()
    total = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, clip in enumerate(loader):
        clip = clip.to(device, non_blocking=True)  # [B,C,T,H,W]
        B = clip.shape[0]
        N = T * tokens_per_frame

        token_mask = make_token_mask(
            B=B,
            N=N,
            mask_ratio=mask_ratio,
            mode=mask_mode,
            T=T,
            tokens_per_frame=tokens_per_frame,
            device=device,
        )

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            pred, target = model(clip, token_mask)
            loss, lstat = total_loss(pred, target, token_mask, cfg.get("mae", {}))

            loss = loss / float(grad_accum)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total += float(loss.item()) * float(grad_accum)

        if (step + 1) % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            d = diag_stats(pred.detach(), target.detach(), token_mask.detach())
            msg = (
                f"[MAE] ep={epoch} step={step+1}/{len(loader)} "
                f"lr={lr_now:.3e} "
                f"loss={total/(step+1):.4f} "
                f"mae={lstat.get('loss_mae', 0.0):.4f} "
                f"mask={d['mask_ratio_actual']:.2f} "
                f"pred_std={d['pred_std']:.3e} tgt_std={d['tgt_std']:.3e} "
                f"pred_abs={d['pred_abs']:.3e} tgt_abs={d['tgt_abs']:.3e}"
            )
            logger.write(msg)

    dt = time.time() - t0
    logger.write(f"[MAE] Epoch done. ep={epoch} avg_loss={total/max(1,len(loader)):.4f} time={dt:.1f}s")
