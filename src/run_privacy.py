import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, ensure_dir, write_csv
from datasets.loader import LazyFrameDataset
from models.mobilevit import build_mobilevit_s
from privacy.visual_mask import YuNetFaceDetector, VisualAnonymizer
from privacy.feature_noise import add_gaussian_noise, apply_feature_mask
from privacy.metrics_privacy import prediction_entropy, privacy_exposure_rate, top1_accuracy
from privacy.attacker import FeatureAttacker


class VideoClassifier(nn.Module):
    """Same definition as train_finetune.py (kept local to avoid imports)."""
    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.backbone = build_mobilevit_s(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [B, C, T, H, W]
        B, C, T, H, W = clip.shape
        feats = []
        for t in range(T):
            _, emb = self.backbone(clip[:, :, t, :, :])
            feats.append(emb)
        feats = torch.stack(feats, dim=1)       # [B, T, D]
        video_emb = feats.mean(dim=1)           # [B, D]
        return self.classifier(video_emb)

    @torch.no_grad()
    def extract_video_embedding(self, clip: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = clip.shape
        feats = []
        for t in range(T):
            _, emb = self.backbone(clip[:, :, t, :, :])
            feats.append(emb)
        feats = torch.stack(feats, dim=1)
        return feats.mean(dim=1)


def maybe_download_yunet(model_path: Path, url: str) -> None:
    """Auto-download YuNet ONNX if missing."""
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] YuNet model missing, downloading to: {model_path}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
        print("[INFO] YuNet download finished.")
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Failed to download YuNet.\n"
            f"Please download manually and place at: {model_path}\n"
            f"Error: {e}"
        )


def scan_images(root: Path, max_images: int, seed: int) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png")
    if not root.exists():
        raise RuntimeError(f"[ERROR] visual_privacy.frame_root not found: {root}")

    imgs = []
    for ext in exts:
        imgs.extend(root.rglob(f"*{ext}"))
    imgs = sorted(imgs)

    if len(imgs) == 0:
        raise RuntimeError(f"[ERROR] No images found under: {root}")

    rng = random.Random(seed)
    if max_images is not None and len(imgs) > max_images:
        imgs = rng.sample(imgs, k=max_images)
    return imgs


def save_visual_examples(pairs: List[Tuple[np.ndarray, np.ndarray]], out_path: Path, cols: int = 4) -> None:
    """Save a grid figure: upper half = before, lower half = after."""
    if len(pairs) == 0:
        return

    h, w = pairs[0][0].shape[:2]
    resized = []
    for a, b in pairs:
        a2 = cv2.resize(a, (w, h))
        b2 = cv2.resize(b, (w, h))
        resized.append((a2, b2))

    rows = []
    for i in range(0, len(resized), cols):
        chunk = resized[i:i + cols]
        row_before = np.concatenate([p[0] for p in chunk], axis=1)
        row_after = np.concatenate([p[1] for p in chunk], axis=1)
        rows.append(np.concatenate([row_before, row_after], axis=0))

    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[INFO] Saved visual examples: {out_path}")


def run_visual_privacy(base: Dict, cfg: Dict, save_dir: Path) -> Path:
    vp = cfg.get("visual_privacy", {})
    if not vp.get("enabled", True):
        print("[INFO] visual_privacy disabled -> skip")
        return save_dir / "visual_privacy.csv"

    frame_root = Path(vp.get("frame_root", Path(base["paths"]["data_root"]) / "FaceForensics_frames"))
    max_images = int(vp.get("max_images", 2000))
    seed = int(base.get("seed", 42))

    images = scan_images(frame_root, max_images=max_images, seed=seed)
    print(f"[INFO] Visual privacy: {len(images)} sampled frames from {frame_root}")

    yunet_path = Path(vp.get("yunet_model", "assets/yunet.onnx"))
    yunet_url = vp.get(
        "yunet_url",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    )
    maybe_download_yunet(yunet_path, yunet_url)

    detector = YuNetFaceDetector(
        model_path=str(yunet_path),
        conf_th=float(vp.get("conf_threshold", 0.6)),
        nms_th=float(vp.get("nms_threshold", 0.3)),
    )
    anonymizer = VisualAnonymizer(
        detector=detector,
        method=vp.get("method", "face_blur"),
        blur_kernel=int(vp.get("blur_kernel", 31)),
    )

    overwrite = bool(vp.get("overwrite", False))
    out_suffix = vp.get("output_suffix", "_frames_privacy")
    out_root = frame_root.parent / (frame_root.name + out_suffix)

    total_frames = 0
    frames_with_face_before = 0
    frames_with_face_after = 0
    faces_before_total = 0
    faces_after_total = 0

    example_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    n_examples = int(vp.get("save_examples", 8))

    t0 = time.time()
    for p in images:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        total_frames += 1

        before_faces = detector.detect(img_bgr)
        n_before = len(before_faces)
        if n_before > 0:
            frames_with_face_before += 1
        faces_before_total += n_before

        anon_bgr, _ = anonymizer.apply(img_bgr)

        after_faces = detector.detect(anon_bgr)
        n_after = len(after_faces)
        if n_after > 0:
            frames_with_face_after += 1
        faces_after_total += n_after

        if overwrite:
            rel = p.relative_to(frame_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), anon_bgr)

        if len(example_pairs) < n_examples and n_before > 0:
            example_pairs.append((img_bgr, anon_bgr))

    dt = time.time() - t0
    if total_frames == 0:
        raise RuntimeError("[ERROR] No valid frames read in visual privacy eval.")

    before_rate = frames_with_face_before / max(1, total_frames)
    after_rate = frames_with_face_after / max(1, total_frames)

    # leakage proxy: conditional detection after anonymization given it was detectable before
    flr_conditional = frames_with_face_after / max(1, frames_with_face_before)
    per_relative = privacy_exposure_rate(before_rate, after_rate)

    rows = [{
        "frame_root": str(frame_root),
        "total_frames": int(total_frames),
        "frames_with_face_before": int(frames_with_face_before),
        "frames_with_face_after": int(frames_with_face_after),
        "avg_faces_before": round(faces_before_total / max(1, total_frames), 6),
        "avg_faces_after": round(faces_after_total / max(1, total_frames), 6),
        "face_frame_rate_before": round(before_rate, 6),
        "face_frame_rate_after": round(after_rate, 6),
        "flr_conditional": round(flr_conditional, 6),
        "per_relative": round(per_relative, 6),
        "seconds": round(dt, 3),
        "overwrite_saved_root": str(out_root) if overwrite else "",
    }]
    out_csv = save_dir / "visual_privacy.csv"
    write_csv(out_csv, list(rows[0].keys()), rows)
    print(f"[INFO] Saved visual privacy CSV: {out_csv}")

    if len(example_pairs) > 0:
        out_img = save_dir / "visual_privacy_examples.jpg"
        save_visual_examples(example_pairs, out_img, cols=4)

    return out_csv


def run_feature_privacy(base: Dict, cfg: Dict, device: torch.device, save_dir: Path) -> Path:
    fp = cfg.get("feature_privacy", {})
    if not fp.get("enabled", True):
        print("[INFO] feature_privacy disabled -> skip")
        return save_dir / "feature_privacy.csv"

    split = Path(base["paths"]["split_root"]) / cfg["dataset"]["split"]
    ds = LazyFrameDataset(
        split_file=str(split),
        mode="supervised",
        clip_len=base["dataset"]["clip_len"],
        stride=base["dataset"]["stride"],
        image_size=base["dataset"]["image_size"],
        seed=base["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=int(base["device"]["num_workers"]),
        pin_memory=bool(base["device"]["pin_memory"]),
    )

    model = VideoClassifier(
        num_classes=int(cfg["dataset"]["num_classes"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
    ).to(device)

    ckpt_path = Path(cfg["model"]["finetune_ckpt"])
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    model.load_state_dict(state, strict=False)
    model.eval()

    # -------- 1️⃣ 先提取 clean embedding（不需要梯度）--------
    zs_all, ys_all, logits_clean_all = [], [], []
    with torch.no_grad():
        for clip, label in loader:
            clip = clip.to(device)
            label = label.to(device)

            z = model.extract_video_embedding(clip)
            logits = model.classifier(z)

            zs_all.append(z.cpu())
            ys_all.append(label.cpu())
            logits_clean_all.append(logits.cpu())

    zs_all = torch.cat(zs_all)
    ys_all = torch.cat(ys_all)
    logits_clean_all = torch.cat(logits_clean_all)

    clean_top1 = top1_accuracy(logits_clean_all, ys_all)
    clean_ent = prediction_entropy(logits_clean_all)
    print(f"[INFO] Clean embeddings -> Top-1={clean_top1:.4f}, Entropy={clean_ent:.4f}")

    sigmas = fp["noise_sigmas"]
    mask_ratios = fp["mask_ratios"]
    attacker_epochs = int(fp.get("attacker_epochs", 10))
    attacker_lr = float(fp.get("attacker_lr", 1e-3))

    rows = []

    # -------- 2️⃣ 对每个隐私强度训练 attacker（需要梯度）--------
    for sigma in sigmas:
        for mask_ratio in mask_ratios:
            z = zs_all.to(device)
            y = ys_all.to(device)

            z_priv = add_gaussian_noise(z, float(sigma))
            z_priv = apply_feature_mask(z_priv, float(mask_ratio))

            # utility
            with torch.no_grad():
                logits = model.classifier(z_priv)
                top1 = top1_accuracy(logits, y)
                top5 = (logits.topk(5, dim=1).indices == y.view(-1, 1)).any(dim=1).float().mean().item()
                ent = prediction_entropy(logits)

            # attacker training (THIS NEEDS GRAD)
            attacker = FeatureAttacker(z_priv.shape[1], int(cfg["dataset"]["num_classes"])).to(device)
            opt = torch.optim.Adam(attacker.parameters(), lr=attacker_lr)
            ce = nn.CrossEntropyLoss()

            attacker.train()
            for _ in range(attacker_epochs):
                opt.zero_grad()
                pred = attacker(z_priv.detach())
                loss = ce(pred, y)
                loss.backward()
                opt.step()

            attacker.eval()
            with torch.no_grad():
                attacker_logits = attacker(z_priv)
                attacker_top1 = top1_accuracy(attacker_logits, y)

            per = privacy_exposure_rate(clean_top1, attacker_top1)

            rows.append({
                "sigma": float(sigma),
                "mask_ratio": float(mask_ratio),
                "top1": round(top1, 6),
                "top5": round(top5, 6),
                "entropy": round(ent, 6),
                "attacker_top1": round(attacker_top1, 6),
                "per_vs_clean": round(per, 6),
            })

            print(
                f"[INFO] sigma={sigma} mask={mask_ratio} | "
                f"top1={top1:.4f} top5={top5:.4f} | "
                f"attacker={attacker_top1:.4f} | ent={ent:.4f}"
            )

    out_csv = save_dir / "feature_privacy.csv"
    write_csv(out_csv, ["sigma", "mask_ratio", "top1", "top5", "entropy", "attacker_top1", "per_vs_clean"], rows)
    print(f"[INFO] Saved feature privacy CSV: {out_csv}")
    return out_csv



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/privacy.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    base = load_config("configs/base.yaml")
    cfg = load_config(args.config)

    device = torch.device("cuda" if (torch.cuda.is_available() and bool(base["device"]["use_cuda"])) else "cpu")
    print(f"[INFO] Device: {device}")

    save_dir = ensure_dir(cfg["output"]["save_dir"])

    # 4.4.1 Visual privacy (FaceForensics++)
    run_visual_privacy(base, cfg, save_dir)

    # 4.4.2 Feature privacy (UCF101)
    run_feature_privacy(base, cfg, device, save_dir)

    print("[INFO] Privacy evaluation finished.")


if __name__ == "__main__":
    main()
