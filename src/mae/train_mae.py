import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .utils import set_seed, load_yaml, ensure_dir, Logger, dump_config, save_checkpoint, keep_last_n_checkpoints
from .engine import build_scheduler, train_one_epoch

from src.datasets.mae_dataset import MAEVideoFramesDataset
from src.models.tinyvit_mae import TinyViTMAE


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    ocfg = cfg["training"]["optimizer"]
    name = str(ocfg.get("name", "adamw")).lower()
    lr = float(ocfg.get("lr", 4e-4))
    wd = float(ocfg.get("weight_decay", 0.05))
    betas = ocfg.get("betas", [0.9, 0.95])

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    results_dir = cfg["paths"]["results_dir"]
    log_dir = os.path.join(results_dir, "logs")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    recon_dir = os.path.join(results_dir, "recon_samples")

    ensure_dir(results_dir)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(recon_dir)

    # log + config dump
    logger = Logger(log_file=os.path.join(log_dir, "mae_train.log"))
    dump_config(cfg, os.path.join(log_dir, "mae_train_used.yaml"))

    # dataset
    split_root = Path(cfg["paths"]["split_root"])
    train_split = split_root / cfg["dataset"]["train_split"]

    ds = MAEVideoFramesDataset(
        split_file=str(train_split),
        clip_len=int(cfg["dataset"]["clip_len"]),
        stride=int(cfg["dataset"]["stride"]),
        image_size=int(cfg["dataset"]["image_size"]),
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=bool(cfg["training"]["pin_memory"]),
        drop_last=True,
    )

    # model
    mcfg = cfg["model"]
    model = TinyViTMAE(
        image_size=int(cfg["dataset"]["image_size"]),
        clip_len=int(cfg["dataset"]["clip_len"]),
        embed_dim=int(mcfg["embed_dim"]),
        encoder_layers=int(mcfg["encoder_layers"]),
        encoder_heads=int(mcfg["encoder_heads"]),
        encoder_mlp_ratio=float(mcfg["encoder_mlp_ratio"]),
        decoder_dim=int(mcfg["decoder_dim"]),
        decoder_layers=int(mcfg["decoder_layers"]),
        decoder_heads=int(mcfg["decoder_heads"]),
        drop=float(mcfg.get("drop", 0.0)),
        attn_drop=float(mcfg.get("attn_drop", 0.0)),
        stage4_pool=int(mcfg.get("stage4_pool", 3)),
    ).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loader))

    amp_enabled = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"]["save_every"])
    keep_last = int(cfg["training"].get("keep_last", 3))

    logger.write(f"[MAE] device={device} seed={seed} epochs={epochs} bs={cfg['training']['batch_size']}")
    logger.write(f"[MAE] train_split={train_split} size={cfg['dataset']['image_size']} clip_len={cfg['dataset']['clip_len']} stride={cfg['dataset']['stride']}")
    logger.write(f"[MAE] mask_ratio={cfg['mae']['mask_ratio']} mask_mode={cfg['mae']['mask_mode']} stage4_pool={cfg['model'].get('stage4_pool', 3)}")

    for ep in range(1, epochs + 1):
        logger.write(f"[MAE] Epoch {ep}/{epochs} started")
        train_one_epoch(
            epoch=ep,
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cfg=cfg,
            logger=logger,
        )

        if (ep % save_every) == 0 or ep == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"mae_ep{ep}.pth")
            save_checkpoint(
                save_path=ckpt_path,
                epoch=ep,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                extra={"cfg": cfg},
            )
            keep_last_n_checkpoints(ckpt_dir, keep_last=keep_last)
            logger.write(f"[MAE] saved: {ckpt_path}")


if __name__ == "__main__":
    main()
