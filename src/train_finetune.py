import os
import sys
from pathlib import Path
import time
import argparse

import torch
import torch.nn as nn
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
        feats = torch.stack(feats, dim=1)   # [B, T, D]
        video_emb = feats.mean(dim=1)       # temporal average
        logits = self.classifier(video_emb)
        return logits


def load_pretrained_ssl(model, ckpt_path):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print("[INFO] No SSL checkpoint loaded")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Only load backbone weights: encoder.* -> backbone.*
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
    return True


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


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


def build_optimizer(ft_cfg, model, mode):
    """
    Keep your original AdamW default behavior, but add optional param-group LRs for two-stage.
    """
    wd = float(ft_cfg["training"]["weight_decay"])

    # If two_stage, use head_lr/backbone_lr when both exist
    if mode == "two_stage":
        head_lr = float(ft_cfg["training"].get("head_lr", ft_cfg["training"]["learning_rate"]))
        backbone_lr = float(ft_cfg["training"].get("backbone_lr", ft_cfg["training"]["learning_rate"]))

        params = []
        # classifier always trainable
        params.append({"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": wd})

        # backbone might be frozen in stage1; only include trainable params
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if len(bb_params) > 0:
            params.append({"params": bb_params, "lr": backbone_lr, "weight_decay": wd})

        opt = torch.optim.AdamW(params)
        print(f"[INFO] Optimizer(two_stage): head_lr={head_lr}, backbone_lr={backbone_lr}, wd={wd}")
        return opt

    # default single-lr optimizer (original behavior)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(ft_cfg["training"]["learning_rate"]),
        weight_decay=wd,
    )
    print(f"[INFO] Optimizer: lr={ft_cfg['training']['learning_rate']}, wd={wd}")
    return opt


def resolve_mode(ft_cfg, cli_mode):
    """
    Four modes:
      - ft_random: random init baseline (no SSL load), full finetune
      - linear_probe: SSL load + freeze backbone for all epochs
      - ft_ssl: SSL load + full finetune
      - two_stage: SSL load + freeze backbone for first stage1_epochs, then unfreeze
    """
    mode = cli_mode or ft_cfg.get("experiment", {}).get("mode", "ft_ssl")
    valid = {"ft_random", "linear_probe", "ft_ssl", "two_stage"}
    if mode not in valid:
        raise ValueError(f"[ERROR] Unknown mode={mode}, must be one of {sorted(list(valid))}")
    return mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--mode", type=str, default=None,
                        help="Override experiment mode: ft_random | linear_probe | ft_ssl | two_stage")
    args = parser.parse_args()

    base_cfg = load_config("configs/base.yaml")
    ft_cfg = load_config(args.config)

    mode = resolve_mode(ft_cfg, args.mode)

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if base_cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Finetune mode: {mode}")

    # paths (keep your original base.yaml behavior)
    split_root = Path(base_cfg["paths"]["split_root"])
    train_split = split_root / ft_cfg["dataset"]["train_split"]
    val_split = split_root / ft_cfg["dataset"]["val_split"]

    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"finetune_{mode}.log"

    # Dataset (keep as-is)
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

    # ---------- mode: init / ssl load ----------
    if mode == "ft_random":
        print("[INFO] ft_random: skip SSL loading (random init)")
    else:
        ok = load_pretrained_ssl(model, ft_cfg["model"].get("pretrained_ssl"))
        if not ok:
            raise RuntimeError("[ERROR] SSL mode selected but pretrained_ssl checkpoint not found!")

    # ---------- mode: freeze settings ----------
    # keep your config field, but mode overrides it safely
    freeze_backbone = bool(ft_cfg["model"].get("freeze_backbone", False))
    two_stage = False
    stage1_epochs = int(ft_cfg["training"].get("stage1_epochs", 0))

    if mode == "linear_probe":
        freeze_backbone = True
        two_stage = False
    elif mode == "ft_ssl":
        freeze_backbone = False
        two_stage = False
    elif mode == "ft_random":
        freeze_backbone = False
        two_stage = False
    elif mode == "two_stage":
        freeze_backbone = True
        two_stage = True
        if stage1_epochs <= 0:
            raise ValueError("[ERROR] two_stage requires training.stage1_epochs > 0")

    if freeze_backbone:
        set_requires_grad(model.backbone, False)
        print("[INFO] Backbone frozen")
    else:
        set_requires_grad(model.backbone, True)

    # Optimizer / AMP
    optimizer = build_optimizer(ft_cfg, model, mode)
    scaler = torch.cuda.amp.GradScaler(enabled=ft_cfg["training"]["amp"] and device.type == "cuda")

    epochs = int(ft_cfg["training"]["epochs"])
    topk = tuple(ft_cfg["evaluation"]["topk"])

    # save_dir per mode to avoid overwriting
    base_save_dir = Path(ft_cfg["paths"]["save_dir"])
    save_dir = base_save_dir / mode
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Start finetuning for {epochs} epochs, save_dir={save_dir}")

    with open(log_path, "a", encoding="utf-8") as log_f:
        best_top1 = 0.0
        for epoch in range(1, epochs + 1):

            # two-stage: unfreeze at stage boundary
            if two_stage and epoch == stage1_epochs + 1:
                msg = "[INFO] two_stage: unfreeze backbone and rebuild optimizer"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
                set_requires_grad(model.backbone, True)
                optimizer = build_optimizer(ft_cfg, model, mode)

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
