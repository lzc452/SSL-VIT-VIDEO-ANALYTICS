import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from datasets.loader import VideoClipDataset
from models.backbone_mobilevit import MobileViTS
from models.heads import ClassificationHead
from privacy.feature_noise import (
    add_gaussian_noise,
    apply_feature_mask,
    compute_feature_leakage,
    compute_entropy,
)


# ---------------------------------------------------------
# Utility
# 支持两种隐私策略：
# 视觉匿名化 → PER
# 特征隐私 → FLR + entropy
# ---------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fix_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_clip_and_get_feat(backbone, clips):
    """
    输入 clips: [B, C, T, H, W]
    输出:
        feats: [B, D] clip-level feature
        frame_feats: [B, T, D] frame-level feature
    """
    B, C, T, H, W = clips.shape
    device = clips.device

    # [B, T, C, H, W]
    clips = clips.permute(0, 2, 1, 3, 4).contiguous()
    frames = clips.view(B * T, C, H, W)

    feat_map = backbone(frames)          # [B*T, D, h, w]
    feat_vec = feat_map.mean(dim=[2, 3]) # [B*T, D]

    frame_feats = feat_vec.view(B, T, -1)
    clip_feats = frame_feats.mean(dim=1)  # [B, D]

    return feat_map, clip_feats


# ---------------------------------------------------------
# Visual Privacy (face blur / mosaic)
# ---------------------------------------------------------
def run_visual_privacy(cfg):
    from privacy.visual_mask import anonymize_frames  # 你已有的 detection+blur 实现

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / cfg["output"]["summary_csv"]

    print("[INFO] Loading dataset for visual privacy...")
    dataset = VideoClipDataset(
        cfg["dataset"]["val_split"],
        mode="supervised",
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    print("[INFO] Loading backbone & head...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MobileViTS().to(device)
    head = ClassificationHead(
        embed_dim=backbone.embed_dim,
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    ckpt = cfg["model"]["finetune_checkpoint"]
    if ckpt and Path(ckpt).exists():
        print(f"[INFO] Loading finetuned weights: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        backbone.load_state_dict(state["backbone"], strict=False)
        head.load_state_dict(state["head"], strict=False)
    else:
        print("[ERROR] finetune checkpoint invalid, abort.")
        return

    method = cfg["visual"]["method"]
    blur_ksize = cfg["visual"]["blur_ksize"]
    blur_sigma = cfg["visual"]["blur_sigma"]
    mosaic_size = cfg["visual"]["mosaic_size"]

    print("[INFO] Running visual anonymization evaluation...")

    total = 0
    clean_correct = 0
    anon_correct = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        B = clips.size(0)
        total += B

        # baseline
        with torch.no_grad():
            _, clip_feat = encode_clip_and_get_feat(backbone, clips)
            logits = head(clip_feat.unsqueeze(-1).unsqueeze(-1))
            pred = logits.argmax(dim=-1)
            clean_correct += (pred == labels).sum().item()

        # anonymize frames
        clips_anon = anonymize_frames(
            clips,
            method=method,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
            mosaic_size=mosaic_size
        )

        with torch.no_grad():
            _, clip_feat_anon = encode_clip_and_get_feat(backbone, clips_anon)
            logits = head(clip_feat_anon.unsqueeze(-1).unsqueeze(-1))
            pred = logits.argmax(dim=-1)
            anon_correct += (pred == labels).sum().item()

    acc_clean = clean_correct / total
    acc_anon = anon_correct / total

    per = 1 - (acc_anon / acc_clean)

    print(f"[INFO] Clean accuracy: {acc_clean:.4f}")
    print(f"[INFO] After anonymization accuracy: {acc_anon:.4f}")
    print(f"[INFO] PER={per:.4f}")

    with open(summary_csv, "w") as f:
        f.write("acc_clean,acc_anon,PER\n")
        f.write(f"{acc_clean:.4f},{acc_anon:.4f},{per:.4f}\n")

    print(f"[INFO] Saved visual privacy summary to {summary_csv}")


# ---------------------------------------------------------
# Feature-level privacy
# ---------------------------------------------------------
def run_feature_privacy(cfg):
    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / cfg["output"]["summary_csv"]

    print("[INFO] Loading dataset for feature-level privacy...")
    dataset = VideoClipDataset(
        cfg["dataset"]["val_split"],
        mode="supervised",
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = MobileViTS().to(device)
    head = ClassificationHead(
        embed_dim=backbone.embed_dim,
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    ckpt = cfg["model"]["finetune_checkpoint"]
    if ckpt and Path(ckpt).exists():
        print(f"[INFO] Loading finetuned weights: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        backbone.load_state_dict(state["backbone"], strict=False)
        head.load_state_dict(state["head"], strict=False)
    else:
        print("[ERROR] finetuned checkpoint invalid, abort.")
        return

    sigma_values = cfg["feature_privacy"]["sigma_values"]
    mask_ratio = cfg["feature_privacy"]["mask_ratio"]

    print("[INFO] Running feature privacy evaluation...")

    lines = ["sigma,acc,flr,entropy\n"]

    for sigma in sigma_values:
        total = 0
        correct = 0
        flr_list = []
        entropy_list = []

        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feat_map_clean, clip_feat_clean = encode_clip_and_get_feat(backbone, clips)

                # noise + mask
                feat_noisy = add_gaussian_noise(feat_map_clean, sigma=sigma)
                feat_noisy = apply_feature_mask(feat_noisy, mask_ratio)

                # get logits
                feat_vec = feat_noisy.mean(dim=[2, 3])     # [B, D]
                logits = head(feat_vec.unsqueeze(-1).unsqueeze(-1))

                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()

                # metrics
                flr_list.append(compute_feature_leakage(feat_map_clean, feat_noisy))
                entropy_list.append(compute_entropy(logits))

            total += clips.size(0)

        acc = correct / total
        flr = np.mean(flr_list)
        entropy = np.mean(entropy_list)

        print(f"[INFO] sigma={sigma} acc={acc:.4f} flr={flr:.4f} entropy={entropy:.4f}")

        lines.append(f"{sigma},{acc:.4f},{flr:.4f},{entropy:.4f}\n")

    with open(summary_csv, "w") as f:
        f.writelines(lines)

    print(f"[INFO] Saved feature privacy summary to {summary_csv}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if "visual" in cfg:
        run_visual_privacy(cfg)
    elif "feature_privacy" in cfg:
        run_feature_privacy(cfg)
    else:
        print("[ERROR] Unknown privacy config")


if __name__ == "__main__":
    main()
