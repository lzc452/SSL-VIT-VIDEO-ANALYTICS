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
from models.heads import ClassificationHead


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
    将clip编码为clip-level特征:
    输入 clips: [B, C, T, H, W]
    流程: frame-level编码 -> [B, T, D] -> 时间平均 -> [B, D]
    """
    B, C, T, H, W = clips.shape
    device = clips.device

    clips = clips.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    frames = clips.view(B * T, C, H, W)                # [B*T, C, H, W]

    feat_map = backbone(frames)                        # [B*T, D, h, w]
    feat_vec = feat_map.mean(dim=[2, 3])               # [B*T, D]

    feats = feat_vec.view(B, T, -1)                    # [B, T, D]
    clip_feat = feats.mean(dim=1)                      # [B, D]
    return clip_feat


def build_optimizer(params, cfg):
    if cfg["name"].lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
    raise NotImplementedError(f"Optimizer {cfg['name']} not implemented")


def evaluate(model_backbone, model_head, loader, device, amp_enable=True):
    model_backbone.eval()
    model_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=amp_enable):
                clip_feat = encode_clip(model_backbone, clips)
                logits = model_head(clip_feat.unsqueeze(-1).unsqueeze(-1))

            _, pred = logits.max(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total if total > 0 else 0.0


def train_finetune(config_path):
    cfg = load_config(config_path)
    fix_seed(cfg["training"]["seed"])

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading supervised datasets...")
    train_dataset = VideoClipDataset(
        cfg["dataset"]["train_split"],
        mode="supervised",
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"]
    )
    val_dataset = VideoClipDataset(
        cfg["dataset"]["val_split"],
        mode="supervised",
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    print("[INFO] Initializing MobileViT-S backbone...")
    backbone = MobileViTS().to(device)

    ssl_ckpt_path = cfg["model"]["ssl_checkpoint"]
    if ssl_ckpt_path and Path(ssl_ckpt_path).exists():
        print(f"[INFO] Loading SSL checkpoint from {ssl_ckpt_path}")
        state = torch.load(ssl_ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "backbone" in state:
            backbone.load_state_dict(state["backbone"], strict=False)
        else:
            try:
                backbone.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[ERROR] Failed to load SSL checkpoint: {e}")
    else:
        print("[INFO] SSL checkpoint not found or path invalid, training from scratch.")

    print("[INFO] Initializing classification head...")
    num_classes = cfg["dataset"]["num_classes"]
    head = ClassificationHead(backbone.embed_dim, num_classes).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = build_optimizer(params, cfg["optimizer"])
    scaler = GradScaler(enabled=cfg["training"]["amp"])
    criterion = torch.nn.CrossEntropyLoss()

    epochs = cfg["training"]["epochs"]
    log_interval = cfg["training"]["log_interval"]
    eval_interval = cfg["training"]["eval_interval"]
    save_interval = cfg["training"]["save_interval"]

    best_acc = 0.0

    print("[INFO] Start supervised fine-tuning...")
    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        running_loss = 0.0

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=cfg["training"]["amp"]):
                clip_feat = encode_clip(backbone, clips)          # [B, D]
                logits = head(clip_feat.unsqueeze(-1).unsqueeze(-1))
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"[INFO] epoch {epoch} batch {batch_idx} loss={loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch} finished, avg_loss={avg_loss:.4f}")

        if epoch % eval_interval == 0:
            val_acc = evaluate(backbone, head, val_loader, device, cfg["training"]["amp"])
            print(f"[INFO] Validation accuracy after epoch {epoch}: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = output_dir / "finetune_best.pth"
                torch.save(
                    {
                        "backbone": backbone.state_dict(),
                        "head": head.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc,
                    },
                    best_path
                )
                print(f"[INFO] Saved best checkpoint to {best_path}")

        if epoch % save_interval == 0:
            ckpt_path = output_dir / f"finetune_epoch_{epoch}.pth"
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path
            )
            print(f"[INFO] Saved epoch checkpoint to {ckpt_path}")

    print(f"[INFO] Fine-tuning completed. Best validation accuracy: {best_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to finetune config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_finetune(args.config)
