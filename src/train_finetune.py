import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.loader import LazyFrameDataset
from models.mobilevit import build_mobilevit_s
from utils import load_config, set_seed


class VideoClassifier(nn.Module):
    """
    MobileViT-S backbone + linear classification head
    """
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        self.backbone = build_mobilevit_s(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, clip):
        """
        clip: [B, C, T, H, W]
        """
        B, C, T, H, W = clip.shape
        feats = []
        for t in range(T):
            _, emb = self.backbone(clip[:, :, t, :, :])
            feats.append(emb)
        feats = torch.stack(feats, dim=1)  # [B, T, D]
        video_emb = feats.mean(dim=1)       # temporal average
        logits = self.classifier(video_emb)
        return logits


def load_pretrained_ssl(model, ckpt_path):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print("[INFO] No SSL checkpoint loaded")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Only load backbone weights
    backbone_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            backbone_state[k.replace("encoder.", "")] = v

    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[INFO] Loaded SSL backbone weights from {ckpt_path}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")


def accuracy_topk(logits, targets, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[k] = (correct_k / targets.size(0)).item()
    return res


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, log_f):
    model.train()
    ce_loss = nn.CrossEntropyLoss()

    total_loss = 0.0
    t0 = time.time()

    for step, (clip, label) in enumerate(loader):
        clip = clip.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"] and device.type == "cuda"):
            logits = model(clip)
            loss = ce_loss(logits, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if (step + 1) % cfg["training"]["log_interval"] == 0:
            msg = f"[INFO] step={step+1}/{len(loader)} loss={loss.item():.4f}"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

    avg_loss = total_loss / max(1, len(loader))
    dt = time.time() - t0
    msg = f"[INFO] Epoch train finished, avg_loss={avg_loss:.4f}, time={dt:.1f}s"
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()

    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, topk, log_f, split="val"):
    model.eval()

    total = 0
    correct_topk = {k: 0.0 for k in topk}

    for clip, label in loader:
        clip = clip.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(clip)
        acc = accuracy_topk(logits, label, topk=topk)

        bs = label.size(0)
        total += bs
        for k in topk:
            correct_topk[k] += acc[k] * bs

    msg = f"[INFO] {split} results: " + ", ".join(
        [f"Top-{k}: {correct_topk[k] / total:.4f}" for k in topk]
    )
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()

    return {k: correct_topk[k] / total for k in topk}


def save_checkpoint(model, epoch, acc1, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"finetune_epoch_{epoch}_top1_{acc1:.4f}.pth"
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved checkpoint: {path}")


def main():
    base_cfg = load_config("configs/base.yaml")
    ft_cfg = load_config("configs/finetune.yaml")

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if base_cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # paths
    split_root = Path(base_cfg["paths"]["split_root"])
    train_split = split_root / ft_cfg["dataset"]["train_split"]
    val_split = split_root / ft_cfg["dataset"]["val_split"]

    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "finetune.log"

    # Dataset
    train_ds = LazyFrameDataset(
        split_file=str(train_split),
        mode="supervised",
        clip_len=base_cfg["dataset"]["clip_len"],
        stride=base_cfg["dataset"]["stride"],
        image_size=base_cfg["dataset"]["image_size"],
        seed=base_cfg["seed"],
    )
    val_ds = LazyFrameDataset(
        split_file=str(val_split),
        mode="supervised",
        clip_len=base_cfg["dataset"]["clip_len"],
        stride=base_cfg["dataset"]["stride"],
        image_size=base_cfg["dataset"]["image_size"],
        seed=base_cfg["seed"] + 999,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ft_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=base_cfg["device"]["num_workers"],
        pin_memory=base_cfg["device"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ft_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=base_cfg["device"]["num_workers"],
        pin_memory=base_cfg["device"]["pin_memory"],
    )

    # Model
    model = VideoClassifier(
        num_classes=ft_cfg["dataset"]["num_classes"],
        embed_dim=ft_cfg["model"]["embed_dim"],
    ).to(device)

    load_pretrained_ssl(model, ft_cfg["model"].get("pretrained_ssl"))

    if ft_cfg["model"].get("freeze_backbone", False):
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("[INFO] Backbone frozen")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_cfg["training"]["learning_rate"],
        weight_decay=ft_cfg["training"]["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=ft_cfg["training"]["amp"] and device.type == "cuda")

    epochs = ft_cfg["training"]["epochs"]
    topk = tuple(ft_cfg["evaluation"]["topk"])
    save_dir = ft_cfg["paths"]["save_dir"]

    print(f"[INFO] Start finetuning for {epochs} epochs")

    with open(log_path, "a", encoding="utf-8") as log_f:
        best_top1 = 0.0
        for epoch in range(1, epochs + 1):
            msg = f"[INFO] Epoch {epoch}/{epochs} started"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

            train_one_epoch(model, train_loader, optimizer, scaler, device, ft_cfg, log_f)
            acc = evaluate(model, val_loader, device, topk, log_f, split="val")

            if acc[1] > best_top1:
                best_top1 = acc[1]
                save_checkpoint(model, epoch, best_top1, save_dir)


if __name__ == "__main__":
    main()
