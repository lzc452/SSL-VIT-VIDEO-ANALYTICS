import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets.loader import VideoClipDataset
from models.backbone_mobilevit import MobileViTS
from models.heads import SSLMultiTaskHead


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fix_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_clip(backbone, clips):
    """
    将视频clip编码为frame-level特征.
    输入 clips: [B, C, T, H, W]
    输出 feats: [B, T, D]
    """
    B, C, T, H, W = clips.shape
    device = clips.device

    clips = clips.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    frames = clips.view(B * T, C, H, W)                # [B*T, C, H, W]

    feat_map = backbone(frames)                        # [B*T, D, h, w]
    feat_vec = feat_map.mean(dim=[2, 3])               # [B*T, D]

    feats = feat_vec.view(B, T, -1)                    # [B, T, D]
    return feats


def build_optimizer(params, cfg):
    if cfg["name"].lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
    raise NotImplementedError(f"Optimizer {cfg['name']} not implemented")


def train_ssl(config_path):
    cfg = load_config(config_path)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fix_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading SSL dataset...")
    train_dataset = VideoClipDataset(
        cfg["dataset"]["train_split"],
        mode="ssl",
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    print("[INFO] Initializing MobileViT-S backbone...")
    backbone = MobileViTS().to(device)

    print("[INFO] Initializing SSL head...")
    ssl_head = SSLMultiTaskHead(
        embed_dim=backbone.embed_dim,
        mask_ratio=cfg["model"]["ssl_tasks"]["mask_ratio"],
        enable_mfr=cfg["model"]["ssl_tasks"]["enable_mfr"],
        enable_top=cfg["model"]["ssl_tasks"]["enable_top"],
    ).to(device)

    params = list(backbone.parameters()) + list(ssl_head.parameters())
    optimizer = build_optimizer(params, cfg["optimizer"])
    scaler = GradScaler(enabled=cfg["training"]["amp"])

    epochs = cfg["training"]["epochs"]
    log_interval = cfg["training"]["log_interval"]
    save_interval = cfg["training"]["save_interval"]

    print("[INFO] Start SSL pretraining...")
    global_step = 0

    for epoch in range(1, epochs + 1):
        backbone.train()
        ssl_head.train()

        running_loss = 0.0

        for batch_idx, clips in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=cfg["training"]["amp"]):
                feats = encode_clip(backbone, clips)           # [B, T, D]
                loss_dict = ssl_head(feats)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % log_interval == 0:
                print(f"[INFO] epoch {epoch} step {global_step} "
                      f"loss_total={loss.item():.4f} "
                      f"mfr={loss_dict['mfr'].item():.4f} "
                      f"top={loss_dict['top'].item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch} finished, avg_loss={avg_loss:.4f}")

        if epoch % save_interval == 0:
            ckpt_path = output_dir / f"ssl_epoch_{epoch}.pth"
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "ssl_head": ssl_head.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path
            )
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

    print("[INFO] SSL pretraining completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_train.yaml",
        help="Path to SSL training config"
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ssl(args.config)
