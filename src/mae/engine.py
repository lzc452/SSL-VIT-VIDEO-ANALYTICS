# src/mae/engine.py
from __future__ import annotations

import time
from typing import Dict, Any

import torch
import torch.nn as nn

from .masking import get_mask_ratio, make_token_mask, count_masked, count_visible
from .losses import mae_recon_loss, reconstruction_error_stats


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler,
    device: torch.device,
    cfg: Dict[str, Any],
    logger,
) -> Dict[str, float]:
    model.train()

    training = cfg.get("training", {})
    mae_cfg = cfg.get("mae", {})
    model_cfg = cfg.get("model", {})

    log_interval = int(training.get("log_interval", 20))
    grad_clip = float(training.get("grad_clip", 1.0))
    grad_accum = int(training.get("grad_accum", 1))
    amp = bool(training.get("amp", True))

    stage4_pool = int(model_cfg.get("stage4_pool", 3))
    tokens_per_frame = stage4_pool * stage4_pool

    mask_mode = str(mae_cfg.get("mask_mode", "tube"))
    schedule = mae_cfg.get("mask_ratio_schedule", [])
    default_mask = float(mae_cfg.get("mask_ratio", 0.8))
    mask_ratio = get_mask_ratio(epoch, schedule, default_mask)

    loss_type = str(mae_cfg.get("loss_type", "l2"))
    normalize_target = bool(mae_cfg.get("normalize_target", True))

    # meters
    n_steps = 0
    loss_sum = 0.0
    l1_sum = 0.0
    l2_sum = 0.0
    pred_std_sum = 0.0
    tgt_std_sum = 0.0

    t0 = time.time()
    data_t = 0.0
    iter_t = 0.0
    last = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, clip in enumerate(loader):
        now = time.time()
        data_t += (now - last)

        clip = clip.to(device, non_blocking=True)  # [C,T,H,W] or [B,C,T,H,W]?
        if clip.dim() == 4:
            clip = clip.unsqueeze(0)
        # ensure [B,C,T,H,W]
        if clip.dim() != 5:
            raise RuntimeError(f"Expected clip dim=5, got {clip.shape}")

        B = clip.size(0)
        T = clip.size(2)

        token_mask = make_token_mask(
            B=B,
            T=T,
            tokens_per_frame=tokens_per_frame,
            mask_ratio=mask_ratio,
            mode=mask_mode,
            device=device,
        )

        with torch.amp.autocast(device_type="cuda", enabled=amp):
            pred, target = model(clip, token_mask=token_mask, stage4_pool=stage4_pool)
            loss = mae_recon_loss(pred, target, loss_type=loss_type, normalize_target=normalize_target)

        if not torch.isfinite(loss):
            logger.write(f"[WARN] epoch={epoch} step={step} loss is not finite: {float(loss)} -> skip step")
            optimizer.zero_grad(set_to_none=True)
            last = time.time()
            continue

        # gradient accumulation
        loss_scaled = loss / float(grad_accum)

        if scaler is not None and amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        do_step = ((step + 1) % grad_accum == 0)
        if do_step:
            if grad_clip > 0:
                if scaler is not None and amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None and amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # stats
        stats = reconstruction_error_stats(pred, target)
        loss_sum += float(loss.detach().item())
        l1_sum += float(stats["l1"])
        l2_sum += float(stats["l2"])
        pred_std_sum += float(stats["pred_std"])
        tgt_std_sum += float(stats["target_std"])
        n_steps += 1

        iter_t += (time.time() - now)
        last = time.time()

        if (step % log_interval) == 0:
            lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"ep={epoch} step={step}/{len(loader)} "
                f"lr={lr:.6g} "
                f"loss={float(loss.detach().item()):.4f} "
                f"l1={stats['l1']:.4f} l2={stats['l2']:.4f} "
                f"std_pred={stats['pred_std']:.3f} std_tgt={stats['target_std']:.3f} "
                f"mask={mask_ratio:.2f} tokens(vis={count_visible(token_mask)},mask={count_masked(token_mask)}) "
                f"t_data={data_t:.2f}s t_iter={iter_t:.2f}s"
            )
            # collapse hint
            if stats["pred_std"] < 0.05:
                msg += " [WARN:pred_std_low]"
            logger.write(msg)
            data_t = 0.0
            iter_t = 0.0

    t1 = time.time()
    if n_steps == 0:
        return {"loss": 1e9, "l1": 1e9, "l2": 1e9, "pred_std": 0.0, "target_std": 0.0, "mask_ratio": mask_ratio, "time_sec": t1 - t0}

    return {
        "loss": loss_sum / n_steps,
        "l1": l1_sum / n_steps,
        "l2": l2_sum / n_steps,
        "pred_std": pred_std_sum / n_steps,
        "target_std": tgt_std_sum / n_steps,
        "mask_ratio": mask_ratio,
        "time_sec": t1 - t0,
    }
