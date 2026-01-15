import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.loader import LazyFrameDataset
from utils import load_config, set_seed
from models.mobilevit import build_mobilevit_s


# ============================================================
# Utils
# ============================================================
def build_masks(B, T, ratio, device):
    n = max(1, int(T * ratio))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:n]
        mask[b, idx] = True
    return mask


def top_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ============================================================
# Model
# ============================================================
class TemporalSSL(nn.Module):
    def __init__(self, embed_dim=256, clip_len=32):
        super().__init__()

        self.encoder = build_mobilevit_s(embed_dim=embed_dim)

        self.pos = nn.Parameter(torch.zeros(1, clip_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc = nn.TransformerEncoderLayer(
            embed_dim, 4, embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(enc, 4)

        # ---- MFM head ----
        self.mfm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- TOP head ----
        self.top_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4)
        )

    def forward(self, clip, mask=None):
        B, C, T, H, W = clip.shape
        x = clip.view(B * T, C, H, W)
        _, feat = self.encoder(x)
        feat = feat.view(B, T, -1)
        feat = feat + self.pos[:, :T]

        if mask is not None:
            feat = feat.clone()
            feat[mask] = 0.0

        ctx = self.temporal(feat)
        ctx = torch.clamp(ctx, -5, 5)
        return feat, ctx


# ============================================================
# TOP augmentation (4-way, balanced)
# ============================================================
def apply_top(clip):
    B, C, T, H, W = clip.shape
    labels = torch.randint(0, 4, (B,), device=clip.device)
    out = clip.clone()
    L = T // 4

    for b in range(B):
        chunks = [clip[b:b+1, :, i*L:(i+1)*L] for i in range(4)]
        if labels[b] == 1:
            order = [0, 2, 1, 3]
        elif labels[b] == 2:
            order = [3, 2, 1, 0]
        elif labels[b] == 3:
            order = [3, 0, 1, 2]
        else:
            continue
        out[b:b+1] = torch.cat([chunks[i] for i in order], dim=2)
    return out, labels


# ============================================================
# Train
# ============================================================
def train_one_epoch(model, loader, opt, device, cfg, epoch):
    model.train()

    mfm_w = cfg["ssl"]["mfm_weight"]
    top_w = cfg["ssl"]["top_weight"]

    total_mfm, total_top = 0, 0
    total_acc = 0

    for clip in loader:
        clip = clip.to(device)
        B, C, T, H, W = clip.shape

        # -------- MFM --------
        mask = build_masks(B, T, cfg["ssl"]["mask_ratio"], device)
        feat, ctx = model(clip, mask)
        pred = model.mfm_head(ctx)[mask]
        tgt = feat.detach()[mask]

        mfm_loss = (
            F.mse_loss(pred, tgt) +
            (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
        )

        # -------- TOP --------
        clip_top, labels = apply_top(clip)
        _, ctx_top = model(clip_top)
        pooled = ctx_top.mean(dim=1)
        logits = model.top_head(pooled)
        top_loss = F.cross_entropy(logits, labels)
        acc = top_accuracy(logits, labels)

        loss = mfm_w * mfm_loss + top_w * top_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_mfm += mfm_loss.item()
        total_top += top_loss.item()
        total_acc += acc

    n = len(loader)
    print(
        f"[EP {epoch}] "
        f"mfm={total_mfm/n:.4f} "
        f"top={total_top/n:.4f} "
        f"top_acc={total_acc/n:.4f}"
    )


# ============================================================
# Main
# ============================================================
def main():
    cfg = load_config("configs/ssl_train.yaml")
    base = load_config("configs/base.yaml")
    cfg = {**base, **cfg}

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LazyFrameDataset(
        split_file=cfg["dataset"]["train_split"],
        mode="ssl",
        clip_len=cfg["dataset"]["clip_len"],
        stride=cfg["dataset"]["stride"],
        image_size=cfg["dataset"]["image_size"],
        seed=cfg["seed"],
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    model = TemporalSSL(
        embed_dim=cfg["model"]["embed_dim"],
        clip_len=cfg["dataset"]["clip_len"]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    for ep in range(1, cfg["training"]["epochs"] + 1):
        train_one_epoch(model, loader, opt, device, cfg, ep)


if __name__ == "__main__":
    main()
