import os
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.loader import VideoClipDataset
from models.backbone_mobilevit import MobileViTS
from models.heads import ClassificationHead
from models.dynamic_infer import encode_frames_to_logits, temporal_dynamic_exit


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fix_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dynamic(config_path):
    cfg = load_config(config_path)
    fix_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / cfg["output"]["summary_csv"]

    print("[INFO] Loading validation dataset for dynamic inference...")
    val_dataset = VideoClipDataset(
        cfg["dataset"]["val_split"],
        mode="supervised",
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    print("[INFO] Initializing MobileViT-S backbone and classification head...")
    backbone = MobileViTS().to(device)
    head = ClassificationHead(
        embed_dim=backbone.embed_dim,
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    ckpt_path = cfg["model"]["finetune_checkpoint"]
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[INFO] Loading finetune checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "backbone" in state and "head" in state:
            backbone.load_state_dict(state["backbone"], strict=False)
            head.load_state_dict(state["head"], strict=False)
        else:
            print("[ERROR] Finetune checkpoint format not as expected.")
    else:
        print("[ERROR] Finetune checkpoint not found, aborting dynamic inference.")
        return

    thresholds = cfg["dynamic"]["thresholds"]
    min_frames = cfg["dynamic"]["min_frames"]
    latency_per_frame_ms = cfg["dynamic"]["approx_latency_per_frame_ms"]
    T = cfg["dataset"]["clip_len"]

    # 统计 baseline 和各个 threshold 下的结果
    total_clips = 0
    baseline_correct = 0

    stats = {
        thr: {
            "correct": 0,
            "sum_frames": 0,
        }
        for thr in thresholds
    }

    print("[INFO] Starting dynamic inference evaluation on validation set...")

    backbone.eval()
    head.eval()

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B = clips.size(0)
            total_clips += B

            # [B, T, num_classes]
            frame_logits = encode_frames_to_logits(backbone, head, clips)

            # baseline: 使用所有 T 帧平均 logits
            avg_logits = frame_logits.mean(dim=1)  # [B, C]
            baseline_pred = avg_logits.argmax(dim=-1)  # [B]
            baseline_correct += (baseline_pred == labels).sum().item()

            # 对每个 clip 逐个阈值做 dynamic policy
            for b in range(B):
                logits_seq = frame_logits[b]  # [T, C]
                label = labels[b].item()

                for thr in thresholds:
                    exit_idx, pred_label, conf = temporal_dynamic_exit(
                        logits_seq, threshold=thr, min_frames=min_frames
                    )
                    # 统计
                    if pred_label == label:
                        stats[thr]["correct"] += 1
                    stats[thr]["sum_frames"] += (exit_idx + 1)

    baseline_acc = baseline_correct / total_clips if total_clips > 0 else 0.0
    print(f"[INFO] Baseline (all frames) accuracy: {baseline_acc:.4f}")

    # 写 CSV: threshold, dynamic_acc, avg_frames, rel_frames, approx_latency_ms
    lines = []
    header = "threshold,dynamic_acc,avg_frames,rel_frames,approx_latency_ms\n"
    lines.append(header)

    for thr in thresholds:
        correct = stats[thr]["correct"]
        sum_frames = stats[thr]["sum_frames"]
        dynamic_acc = correct / total_clips if total_clips > 0 else 0.0
        avg_frames = sum_frames / total_clips if total_clips > 0 else 0.0
        rel_frames = avg_frames / float(T)
        approx_latency = avg_frames * latency_per_frame_ms

        print(
            f"[INFO] threshold={thr:.2f} "
            f"acc={dynamic_acc:.4f} "
            f"avg_frames={avg_frames:.2f} "
            f"rel_frames={rel_frames:.3f} "
            f"approx_latency_ms={approx_latency:.2f}"
        )

        line = f"{thr},{dynamic_acc:.4f},{avg_frames:.2f},{rel_frames:.3f},{approx_latency:.2f}\n"
        lines.append(line)

    with open(summary_csv, "w") as f:
        f.writelines(lines)

    print(f"[INFO] Dynamic inference summary saved to {summary_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dynamic_infer.yaml",
        help="Path to dynamic inference config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dynamic(args.config)
