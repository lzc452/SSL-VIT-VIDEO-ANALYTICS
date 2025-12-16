import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# make imports work when running: python src/train_ssl.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.loader import LazyFrameDataset
from utils import load_config, set_seed
from models.mobilevit import build_mobilevit_s


class TemporalSSLModel(nn.Module):
    """
    Frame encoder (MobileViT) + temporal transformer for SSL objectives:
      - Masked Feature Modeling (MFM) on frame embeddings
      - Temporal Order Prediction (TOP)
    """
    def __init__(self, embed_dim=256, temporal_layers=4, temporal_heads=4):
        super().__init__()
        self.encoder = build_mobilevit_s(embed_dim=embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=temporal_layers)

        # For MFM: predict target embedding
        self.mfm_head = nn.Linear(embed_dim, embed_dim)

        # For TOP: binary classification (correct vs shuffled)
        self.top_head = nn.Linear(embed_dim, 2)

    def encode_frames(self, clip):
        """
        clip: [B, C, T, H, W]
        return: frame_embs [B, T, D]
        """
        B, C, T, H, W = clip.shape
        frame_embs = []
        for t in range(T):
            _, emb = self.encoder(clip[:, :, t, :, :])  # [B, D]
            frame_embs.append(emb)
        x = torch.stack(frame_embs, dim=1)  # [B, T, D]
        return x

    def forward(self, clip, mfm_mask=None):
        """
        clip: [B,C,T,H,W]
        mfm_mask: [B,T] boolean mask, True means masked token
        """
        x = self.encode_frames(clip)  # [B,T,D]
        x_in = x

        if mfm_mask is not None:
            # replace masked tokens with zeros (simple, stable)
            x_in = x_in.clone()
            x_in[mfm_mask] = 0.0

        h = self.temporal(x_in)  # [B,T,D]
        return x, h  # (target embeddings, contextual embeddings)


def build_masks(B, T, mask_ratio, device):
    num_mask = max(1, int(T * mask_ratio))
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:num_mask]
        mask[b, idx] = True
    return mask


def maybe_shuffle_order(clip, prob=0.5):
    """
    For TOP: produce (clip_variant, label)
      label=0: correct order
      label=1: shuffled order
    """
    B, C, T, H, W = clip.shape
    labels = torch.zeros(B, dtype=torch.long, device=clip.device)

    clip_out = clip.clone()
    for b in range(B):
        if torch.rand(1, device=clip.device).item() < prob:
            perm = torch.randperm(T, device=clip.device)
            clip_out[b] = clip_out[b, :, perm, :, :]
            labels[b] = 1
    return clip_out, labels


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, log_f):
    model.train()

    mask_ratio = cfg["ssl_objectives"]["mask_ratio"]
    lam_mfm = cfg["ssl_objectives"]["lambda_mfm"]
    lam_top = cfg["ssl_objectives"]["lambda_top"]
    top_prob = cfg["ssl_objectives"]["top_shuffle_prob"]

    total_loss = 0.0
    total_mfm = 0.0
    total_top = 0.0

    t0 = time.time()
    for step, clip in enumerate(loader):
        clip = clip.to(device, non_blocking=True)

        # TOP branch: maybe shuffle
        clip_top, top_labels = maybe_shuffle_order(clip, prob=top_prob)

        # MFM mask
        B, C, T, H, W = clip.shape
        mfm_mask = build_masks(B, T, mask_ratio, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"] and device.type == "cuda"):
            # MFM forward
            target_emb, ctx_emb = model(clip, mfm_mask=mfm_mask)  # [B,T,D]
            pred = model.mfm_head(ctx_emb)  # [B,T,D]

            # MFM loss computed only on masked positions
            mfm_loss = F.mse_loss(pred[mfm_mask], target_emb.detach()[mfm_mask])

            # TOP forward
            _, ctx_top = model(clip_top, mfm_mask=None)  # [B,T,D]
            pooled = ctx_top.mean(dim=1)  # [B,D]
            logits = model.top_head(pooled)  # [B,2]
            top_loss = F.cross_entropy(logits, top_labels)

            loss = lam_mfm * mfm_loss + lam_top * top_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mfm += mfm_loss.item()
        total_top += top_loss.item()

        if (step + 1) % cfg["training"]["log_interval"] == 0:
            msg = (
                f"[INFO] step={step+1}/{len(loader)} "
                f"loss={loss.item():.4f} mfm={mfm_loss.item():.4f} top={top_loss.item():.4f}"
            )
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

    avg_loss = total_loss / max(1, len(loader))
    avg_mfm = total_mfm / max(1, len(loader))
    avg_top = total_top / max(1, len(loader))
    dt = time.time() - t0

    msg = f"[INFO] Epoch finished, avg_loss={avg_loss:.4f}, avg_mfm={avg_mfm:.4f}, avg_top={avg_top:.4f}, time={dt:.1f}s"
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()

    return avg_loss


def save_checkpoint(model, optimizer, epoch, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"ssl_epoch_{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"[INFO] Saved checkpoint: {ckpt_path}")


def main():
    base_cfg = load_config("configs/base.yaml")
    ssl_cfg = load_config("configs/ssl_train.yaml")
    # merge: ssl_cfg overrides base_cfg keys where needed
    cfg = {**base_cfg, **ssl_cfg}
    cfg["dataset"] = {**base_cfg["dataset"], **ssl_cfg.get("dataset", {})}
    cfg["training"] = {**ssl_cfg.get("training", {})}
    cfg["model"] = {**ssl_cfg.get("model", {})}
    cfg["ssl_objectives"] = {**ssl_cfg.get("ssl_objectives", {})}
    cfg["optimizer"] = {**ssl_cfg.get("optimizer", {})}

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if base_cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # paths
    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_ssl.log"

    split_root = Path(base_cfg["paths"]["split_root"])
    train_split = split_root / cfg["dataset"]["train_split"]

    # Dataset & Loader
    train_ds = LazyFrameDataset(
        split_file=str(train_split),
        mode="ssl",
        clip_len=base_cfg["dataset"]["clip_len"],
        stride=base_cfg["dataset"]["stride"],
        image_size=base_cfg["dataset"]["image_size"],
        seed=base_cfg["seed"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=base_cfg["device"]["num_workers"],
        pin_memory=base_cfg["device"]["pin_memory"],
        drop_last=True,
    )

    # Model
    model = TemporalSSLModel(
        embed_dim=cfg["model"]["embed_dim"],
        temporal_layers=cfg["model"]["temporal_layers"],
        temporal_heads=cfg["model"]["temporal_heads"],
    ).to(device)

    # Optimizer
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"] and device.type == "cuda")

    save_dir = cfg["training"]["save_dir"]
    save_every = int(cfg["training"]["save_every"])
    epochs = int(cfg["training"]["epochs"])

    print(f"[INFO] Start SSL training: epochs={epochs}, batch_size={cfg['training']['batch_size']}")
    print(f"[INFO] Save dir: {save_dir}")

    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(1, epochs + 1):
            msg = f"[INFO] Epoch {epoch}/{epochs} started"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

            train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, log_f)

            if (epoch % save_every) == 0 or epoch == epochs:
                save_checkpoint(model, optimizer, epoch, save_dir)


if __name__ == "__main__":
    main()
