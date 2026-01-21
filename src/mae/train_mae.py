# src/mae/train_mae.py
from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from .utils import set_seed, load_yaml, ensure_dir, Logger, dump_config, save_checkpoint, keep_last_n_checkpoints, seed_worker, make_generator
from .engine import train_one_epoch
from .metrics import update_best, format_metrics

from src.datasets.mae_dataset import MAEVideoDataset
from src.models.tinyvit_mae import TinyViTMAE


def build_model(cfg: Dict[str, Any]) -> TinyViTMAE:
    model_cfg = cfg.get("model", {})
    mae_cfg = cfg.get("mae", {})

    backbone_name = str(model_cfg.get("backbone_name", model_cfg.get("backbone", "tiny_vit_21m_224")))
    pretrained = str(model_cfg.get("pretrained", ""))

    stage4_pool = int(model_cfg.get("stage4_pool", 3))
    embed_dim = int(model_cfg.get("embed_dim", 256))

    decoder_dim = int(model_cfg.get("decoder_dim", 512))
    decoder_depth = int(model_cfg.get("decoder_depth", model_cfg.get("decoder_layers", 2)))
    decoder_heads = int(model_cfg.get("decoder_heads", 8))

    model = TinyViTMAE(
        backbone=backbone_name,
        pretrained=pretrained,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        stage4_pool=stage4_pool,
    )
    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", opt_cfg.get("base_lr", 3e-4)))
    wd = float(opt_cfg.get("weight_decay", 0.05))
    betas = opt_cfg.get("betas", [0.9, 0.95])
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sch_cfg = cfg.get("scheduler", {})
    training = cfg.get("training", {})
    epochs = int(training.get("epochs", 200))
    warmup = int(sch_cfg.get("warmup_epochs", 10))
    min_lr_ratio = float(sch_cfg.get("min_lr_ratio", 0.05))

    def lr_lambda(ep: int):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        # cosine
        t = (ep - warmup) / float(max(1, epochs - warmup))
        import math
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output dirs
    output = cfg.get("output", {})
    out_dir = str(output.get("out_dir", cfg.get("paths", {}).get("results_dir", "results/mae_ssl")))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    vis_dir = os.path.join(out_dir, "vis")

    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(vis_dir)

    logger = Logger(os.path.join(log_dir, "train_mae.log"))
    dump_config(cfg, os.path.join(log_dir, "config_dump.json"))

    # dataset/dataloader
    ds_cfg = cfg.get("dataset", {})
    inp = cfg.get("input", cfg.get("data", {}))
    image_size = int(inp.get("image_size", 112))
    clip_len = int(inp.get("clip_len", 32))
    stride = int(inp.get("stride", 4))
    mean = tuple(inp.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(inp.get("std", [0.229, 0.224, 0.225]))

    train_split = str(ds_cfg.get("train_split", "data/splits/UCF101_train.txt"))
    train_set = MAEVideoDataset(
        split_path=train_split,
        image_size=image_size,
        clip_len=clip_len,
        stride=stride,
        mean=mean,
        std=std,
        training=True,
    )

    dl_cfg = cfg.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 128))
    num_workers = int(dl_cfg.get("num_workers", 8))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 2))
    drop_last = bool(dl_cfg.get("drop_last", True))

    g = make_generator(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # model/opt/sch
    model = build_model(cfg).to(device)

    # log backbone load info if available
    try:
        bi = model.encoder.load_info  # TinyViTBackbone.load_info
        if bi.get("missing") or bi.get("unexpected"):
            logger.write(f"[LOAD] missing_keys={len(bi.get('missing', []))} unexpected_keys={len(bi.get('unexpected', []))}")
    except Exception:
        pass

    n_params = sum(p.numel() for p in model.parameters())
    logger.write(f"[MODEL] params={n_params:,}")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    training = cfg.get("training", {})
    epochs = int(training.get("epochs", 200))
    amp = bool(training.get("amp", True))

    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    # resume
    resume = str(training.get("resume", ""))
    best: Dict[str, Any] = {"loss": 1e9, "epoch": 0}
    start_ep = 1
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best = ckpt.get("best", best)
        start_ep = int(ckpt.get("epoch", 0)) + 1
        logger.write(f"[RESUME] from={resume} start_ep={start_ep} best={best}")

    save_every = int(training.get("save_every", 10))
    keep_last = int(training.get("keep_last", 5))

    vis_cfg = cfg.get("visualize", {})
    vis_every = int(vis_cfg.get("every", 10))

    from .visualize import maybe_save_vis

    for ep in range(start_ep, epochs + 1):
        logger.write(f"[MAE] Epoch {ep}/{epochs} started")
        stats = train_one_epoch(
            epoch=ep,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            logger=logger,
        )
        stats["epoch"] = float(ep)
        best = update_best(best, stats, key="loss", mode="min")
        logger.write(format_metrics(ep, stats, best))

        if (ep % vis_every) == 0:
            maybe_save_vis(ep, model, train_set, device, cfg, out_dir=out_dir, logger=logger)

        if (ep % save_every) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ssl_epoch_{ep:03d}.pth")
            save_checkpoint(ckpt_path, model, optimizer, scaler, scheduler, ep, best, cfg)
            keep_last_n_checkpoints(ckpt_dir, prefix="ssl_epoch_", keep=keep_last)

    logger.write("[DONE] training finished")


if __name__ == "__main__":
    main()
