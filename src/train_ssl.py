import os
import sys
import math
import time
import copy
from pathlib import Path
from typing import Dict, Any, Tuple

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
# Loss helpers
# ============================================================

def cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()

def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # z: [N, D]
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))

@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, m: float):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.data.mul_(m).add_(p.data, alpha=1.0 - m)

def build_mask(B: int, T: int, ratio: float, device) -> torch.Tensor:
    n = max(1, int(T * ratio))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:n]
        mask[b, idx] = True
    return mask


# ============================================================
# TOP permutation: 4-way
# ============================================================

def _perm_index_4way(T: int, label: int) -> torch.Tensor:
    """
    4-way permutation indices for frames:
      0: identity
      1: reverse
      2: swap halves
      3: rotate quarters (Q1 Q2 Q3 Q4 -> Q2 Q3 Q4 Q1)
    """
    idx = torch.arange(T)
    if label == 0:
        return idx
    if label == 1:
        return torch.flip(idx, dims=[0])
    if label == 2:
        half = T // 2
        return torch.cat([idx[half:], idx[:half]], dim=0)
    # label == 3
    q = T // 4
    if q == 0:
        return idx  # fallback if T < 4
    return torch.cat([idx[q:], idx[:q]], dim=0)

def permute_frames_4way(clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    clip: [B, C, T, H, W]
    Returns:
      clip_top: [B, C, T, H, W] permuted per-sample
      labels: [B] in {0,1,2,3}
    """
    B, C, T, H, W = clip.shape
    device = clip.device
    labels = torch.randint(0, 4, (B,), device=device)
    clip_top = clip.clone()
    for b in range(B):
        idx = _perm_index_4way(T, int(labels[b].item())).to(device)
        clip_top[b] = clip[b, :, idx, :, :]
    return clip_top, labels


# ============================================================
# Model
# ============================================================

class TemporalSSL(nn.Module):
    def __init__(self, embed_dim: int, layers: int, heads: int, clip_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_len = clip_len

        self.encoder = build_mobilevit_s(embed_dim=embed_dim)

        self.pos = nn.Parameter(torch.zeros(1, clip_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc, layers)

        # Predictor with BN: expects [N, D]
        self.predictor_fc1 = nn.Linear(embed_dim, embed_dim)
        self.predictor_bn1 = nn.BatchNorm1d(embed_dim)
        self.predictor_fc2 = nn.Linear(embed_dim, embed_dim)

        # TOP head (4-way)
        self.top_head = nn.Linear(embed_dim, 4)

    def predictor(self, x_bt_d: torch.Tensor) -> torch.Tensor:
        """
        x_bt_d: [B*T, D]  -> [B*T, D]
        """
        x = self.predictor_fc1(x_bt_d)
        x = self.predictor_bn1(x)
        x = F.gelu(x)
        x = self.predictor_fc2(x)
        return x

    def forward_tokens(self, clip: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Return temporal context tokens: ctx [B, T, D]
        clip: [B, C, T, H, W]
        mask: [B, T] boolean or None
        """
        B, C, T, H, W = clip.shape
        x = clip.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
        x = x.view(B * T, C, H, W)                    # [B*T,C,H,W]

        # encoder returns (maybe logits, features). keep your original convention:
        _, f = self.encoder(x)                        # f: [B*T, D]
        f = f.view(B, T, -1)                          # [B, T, D]
        f = f + self.pos[:, :T, :]                    # add pos

        if mask is not None:
            f = f.clone()
            f[mask] = self.mask_token.expand(B, T, -1)[mask]

        ctx = self.temporal(f)                        # [B, T, D]
        return ctx


# ============================================================
# Train step
# ============================================================

def train_one_epoch(
    model: TemporalSSL,
    ema: TemporalSSL,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    sch,
    device,
    cfg: Dict[str, Any],
    epoch: int
):
    model.train()
    ema.eval()

    ssl = cfg["ssl_objectives"]
    tr = cfg["training"]
    use_amp = bool(tr.get("amp", True))

    top_enabled = epoch >= int(ssl["top_start_epoch"])
    top_every = int(ssl.get("top_every", 1))
    top_subsample = float(ssl.get("top_subsample", 1.0))
    top_detach_backbone = bool(ssl.get("top_detach_backbone", False))

    running = {
        "mfm": 0.0,
        "var": 0.0,
        "top": 0.0,
        "n": 0
    }

    for step, clip in enumerate(loader, start=1):
        clip = clip.to(device, non_blocking=True)  # [B,C,T,H,W]
        B, C, T, H, W = clip.shape

        mask = build_mask(B, T, ssl["mask_ratio"], device)

        # -------- Teacher target (no grad) --------
        with torch.no_grad():
            # teacher sees full sequence (unmasked)
            ctx_t = ema.forward_tokens(clip, mask=None)           # [B,T,D]
            z_t = ctx_t[mask]                                     # [N_mask, D]
            z_t = F.normalize(z_t, dim=-1)

        # -------- Student forward (masked) --------
        with torch.cuda.amp.autocast(enabled=use_amp):
            ctx_s = model.forward_tokens(clip, mask=mask)         # [B,T,D]
            # predictor expects [B*T,D]
            ctx_s_flat = ctx_s.reshape(B * T, -1)                 # [B*T,D]
            pred_flat = model.predictor(ctx_s_flat)               # [B*T,D]
            pred = pred_flat.reshape(B, T, -1)                    # [B,T,D]
            z_s = pred[mask]                                      # [N_mask,D]
            z_s = F.normalize(z_s, dim=-1)

            loss_mfm = cosine_loss(z_s, z_t)
            loss_var = variance_loss(z_s, ssl["var_target_std"], ssl["var_eps"])
            loss = ssl["mfm_weight"] * loss_mfm + ssl["var_weight"] * loss_var

        # -------- TOP (optional + OOM safeguard) --------
        loss_top = None
        if top_enabled and (step % top_every == 0) and ssl["top_weight"] > 0:
            # subsample batch to reduce memory
            if top_subsample < 1.0:
                k = max(2, int(B * top_subsample))
                idx_b = torch.randperm(B, device=device)[:k]
                clip_top_src = clip[idx_b]
            else:
                idx_b = None
                clip_top_src = clip

            # real permutation + true labels
            clip_top, labels = permute_frames_4way(clip_top_src)

            with torch.cuda.amp.autocast(enabled=use_amp):
                ctx_top = model.forward_tokens(clip_top, mask=None)   # [b',T,D]
                feat = ctx_top.mean(dim=1)                            # [b',D]

                if top_detach_backbone:
                    feat = feat.detach()  # only train top_head, very stable

                logits = model.top_head(feat)                         # [b',4]
                loss_top = F.cross_entropy(logits, labels)
                loss = loss + ssl["top_weight"] * loss_top

            # free TOP tensors ASAP
            del clip_top, labels, ctx_top, feat, logits
            if idx_b is not None:
                del clip_top_src, idx_b

        # -------- Backprop --------
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        if tr.get("clip_grad_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad_norm"])

        scaler.step(opt)
        scaler.update()

        # EMA update
        update_ema(ema, model, ssl["ema_momentum"])

        if sch is not None:
            sch.step()

        # Logging stats
        running["mfm"] += float(loss_mfm.detach().item())
        running["var"] += float(loss_var.detach().item())
        if loss_top is not None:
            running["top"] += float(loss_top.detach().item())
        running["n"] += 1

        # free big tensors
        del ctx_s, ctx_s_flat, pred_flat, pred, z_s, z_t, ctx_t, loss, loss_mfm, loss_var
        # NOTE: do not call empty_cache each step; it slows training.

        if (step % tr["log_interval"]) == 0:
            avg_mfm = running["mfm"] / running["n"]
            avg_var = running["var"] / running["n"]
            avg_top = running["top"] / max(1, running["n"])
            print(
                f"[INFO] ep={epoch} step={step}/{len(loader)} "
                f"mfm={avg_mfm:.4f} var={avg_var:.4f} top={avg_top:.4f} "
                f"(top_on={'Y' if top_enabled else 'N'} every={top_every} subs={top_subsample})"
            )

    # epoch summary
    avg_mfm = running["mfm"] / max(1, running["n"])
    avg_var = running["var"] / max(1, running["n"])
    avg_top = running["top"] / max(1, running["n"])
    print(f"[INFO] Epoch done. ep={epoch} avg_mfm={avg_mfm:.4f} avg_var={avg_var:.4f} avg_top={avg_top:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    cfg = load_config("configs/ssl_train.yaml")
    base = load_config("configs/base.yaml")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LazyFrameDataset(
        split_file=os.path.join(base["paths"]["split_root"], cfg["dataset"]["train_split"]),
        mode="ssl",
        clip_len=base["dataset"]["clip_len"],
        stride=base["dataset"]["stride"],
        image_size=base["dataset"]["image_size"],
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )

    model = TemporalSSL(
        embed_dim=cfg["model"]["embed_dim"],
        layers=cfg["model"]["temporal_layers"],
        heads=cfg["model"]["temporal_heads"],
        clip_len=base["dataset"]["clip_len"],
    ).to(device)

    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)))

    # (Optional) scheduler: keep simple; if you have your own scheduler in repo, plug it here.
    sch = None

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, int(cfg["training"]["epochs"]) + 1):
        train_one_epoch(model, ema, loader, opt, scaler, sch, device, cfg, ep)

        if (ep % int(cfg["training"]["save_every"])) == 0:
            ckpt = {
                "epoch": ep,
                "student": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
            }
            torch.save(ckpt, save_dir / f"ssl_ep{ep}.pth")
            print(f"[INFO] Saved checkpoint: {save_dir / f'ssl_ep{ep}.pth'}")


if __name__ == "__main__":
    main()
