import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.loader import LazyFrameDataset
from models.mobilevit import build_mobilevit_s


class VideoClassifier(nn.Module):
    """
    Same as finetune: per-frame MobileViT embedding + temporal mean + linear head
    """
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        self.backbone = build_mobilevit_s(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, clip):
        B, C, T, H, W = clip.shape
        feats = []
        for t in range(T):
            _, emb = self.backbone(clip[:, :, t, :, :])
            feats.append(emb)
        feats = torch.stack(feats, dim=1)
        video_emb = feats.mean(dim=1)
        return self.classifier(video_emb)


def try_load_ckpt(model, ckpt_path):
    if ckpt_path is None:
        print("[INFO] init_ckpt not provided, using random init")
        return
    if not os.path.isfile(ckpt_path):
        print(f"[INFO] init_ckpt not found: {ckpt_path}, using random init")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded init_ckpt: {ckpt_path}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")


def build_loader(split_file, base_cfg, batch_size, mode="supervised", seed=42, shuffle=True):
    ds = LazyFrameDataset(
        split_file=split_file,
        mode=mode,
        clip_len=base_cfg["dataset"]["clip_len"],
        stride=base_cfg["dataset"]["stride"],
        image_size=base_cfg["dataset"]["image_size"],
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(base_cfg["device"]["num_workers"]),
        pin_memory=bool(base_cfg["device"]["pin_memory"]),
        drop_last=shuffle,
    )
    return ds, loader
