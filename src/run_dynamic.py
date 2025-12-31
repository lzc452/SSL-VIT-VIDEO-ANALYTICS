import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.loader import LazyFrameDataset
from models.mobilevit import build_mobilevit_s
from models.dynamic_infer import streaming_early_exit, select_topk_frames
from utils import load_config, set_seed


class VideoClassifier(nn.Module):
    """
    Same structure as finetune classifier:
    - MobileViT backbone per frame
    - temporal average pooling
    - linear classifier
    """
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        self.backbone = build_mobilevit_s(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, clip):
        # clip: [B, C, T, H, W]
        B, C, T, H, W = clip.shape
        feats = []
        for t in range(T):
            _, emb = self.backbone(clip[:, :, t, :, :])  # [B, D]
            feats.append(emb)
        feats = torch.stack(feats, dim=1)              # [B, T, D]
        video_emb = feats.mean(dim=1)                  # [B, D]
        logits = self.classifier(video_emb)            # [B, K]
        return logits


def load_finetune_ckpt(model, ckpt_path, device):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"[ERROR] finetune_ckpt not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    # accept both "state_dict only" and wrapped dict
    if isinstance(state, dict) and all(k in state for k in ["model", "epoch"]):
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded finetune checkpoint: {ckpt_path}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()


def accuracy_topk(logits, targets, topk=(1, )):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    out = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        out[k] = correct_k / targets.size(0)
    return out


@torch.no_grad()
def run_early_exit(model, loader, device, cfg, log_f):
    thresholds = cfg["dynamic"]["confidence_thresholds"]
    min_frames = int(cfg["dynamic"]["min_frames"])
    max_frames = int(cfg["dynamic"]["max_frames"])
    step = int(cfg["dynamic"]["frame_step"])

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "early_exit_results.csv"

    topk = (1, 5)
    results_lines = ["threshold,top1,top5,avg_frames,avg_conf,avg_latency_ms,throughput_fps\n"]

    num_warmup = int(cfg["runtime"]["num_warmup"])
    num_measure = int(cfg["runtime"]["num_measure"])

    for thr in thresholds:
        correct1 = 0.0
        correct5 = 0.0
        total = 0
        frames_used_sum = 0.0
        conf_sum = 0.0
        latencies = []

        for i, (clip, label) in enumerate(loader):
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B, C, T, H, W = clip.shape
            T = min(T, max_frames)

            do_measure = (i >= num_warmup) and (i < num_warmup + num_measure)

            if do_measure and device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()

            # Streaming early-exit (true compute saving; frames encoded once)
            final_logits, stats = streaming_early_exit(
                backbone=model.backbone,
                classifier=model.classifier,
                clip=clip[:, :, :T, :, :],
                threshold=float(thr),
                min_frames=min_frames,
                max_frames=T,
                frame_step=step,
            )

            if do_measure and device.type == "cuda":
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))  # ms

            acc = accuracy_topk(final_logits, label, topk=topk)
            correct1 += acc[1] * B
            correct5 += acc[5] * B
            total += B

            frames_used_sum += stats.used_frames.float().sum().item()
            conf_sum += stats.final_conf.sum().item()

        top1 = correct1 / total
        top5 = correct5 / total
        avg_frames = frames_used_sum / total
        avg_conf = conf_sum / total

        if len(latencies) > 0:
            avg_latency = sum(latencies) / len(latencies)
            clips_per_sec = (loader.batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0
            throughput_fps = clips_per_sec * avg_frames
        else:
            avg_latency = 0.0
            throughput_fps = 0.0

        line = f"{thr:.2f},{top1:.6f},{top5:.6f},{avg_frames:.3f},{avg_conf:.4f},{avg_latency:.3f},{throughput_fps:.2f}\n"
        results_lines.append(line)

        msg = f"[INFO] EarlyExit thr={thr:.2f} Top1={top1:.4f} Top5={top5:.4f} avg_frames={avg_frames:.2f} avg_latency_ms={avg_latency:.2f}"
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    if cfg["output"]["save_csv"]:
        csv_path.write_text("".join(results_lines), encoding="utf-8")
        print(f"[INFO] Saved early-exit summary CSV: {csv_path}")


@torch.no_grad()
def run_frame_gating(model, loader, device, cfg, log_f):
    topk_list = cfg["dynamic"]["gating_topk_list"]
    score_type = cfg["dynamic"]["gating_score"]

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "frame_gating_results.csv"

    topk_eval = (1, 5)
    num_warmup = int(cfg["runtime"]["num_warmup"])
    num_measure = int(cfg["runtime"]["num_measure"])

    results_lines = ["k,top1,top5,avg_latency_ms,throughput_clips_per_s\n"]

    for k in topk_list:
        correct1 = 0.0
        correct5 = 0.0
        total = 0
        latencies = []

        for i, (clip, label) in enumerate(loader):
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            B, C, T, H, W = clip.shape
            k_eff = min(int(k), T)

            clip_sel, _ = select_topk_frames(clip, k=k_eff, score_type=score_type)

            do_measure = (i >= num_warmup) and (i < num_warmup + num_measure)
            if do_measure and device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()

            logits = model(clip_sel)

            if do_measure and device.type == "cuda":
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))

            acc = accuracy_topk(logits, label, topk=topk_eval)
            correct1 += acc[1] * B
            correct5 += acc[5] * B
            total += B

        top1 = correct1 / total
        top5 = correct5 / total

        if len(latencies) > 0:
            avg_latency = sum(latencies) / len(latencies)
            clips_per_sec = (loader.batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0
        else:
            avg_latency = 0.0
            clips_per_sec = 0.0

        results_lines.append(f"{k_eff},{top1:.6f},{top5:.6f},{avg_latency:.3f},{clips_per_sec:.2f}\n")

        msg = f"[INFO] FrameGating k={k_eff} Top1={top1:.4f} Top5={top5:.4f} avg_latency_ms={avg_latency:.2f}"
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    if cfg["output"]["save_csv"]:
        csv_path.write_text("".join(results_lines), encoding="utf-8")
        print(f"[INFO] Saved frame-gating summary CSV: {csv_path}")


@torch.no_grad()
def run_hybrid(model, loader, device, cfg, log_f):
    """Hybrid dynamic inference: frame gating -> streaming early-exit.

    This is the most "journal-ready" closed-loop setting in this repo:
    - gating reduces the number of frames processed;
    - early-exit stops even earlier when confidence is sufficient;
    - both knobs (k, threshold) produce a dense trade-off frontier.
    """

    topk_list = cfg["dynamic"]["gating_topk_list"]
    score_type = cfg["dynamic"]["gating_score"]
    thresholds = cfg["dynamic"]["confidence_thresholds"]
    min_frames = int(cfg["dynamic"]["min_frames"])
    frame_step = int(cfg["dynamic"]["frame_step"])

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "hybrid_results.csv"

    topk_eval = (1, 5)
    num_warmup = int(cfg["runtime"]["num_warmup"])
    num_measure = int(cfg["runtime"]["num_measure"])

    results_lines = ["k,threshold,top1,top5,avg_used_frames,avg_conf,avg_latency_ms\n"]

    for k in topk_list:
        for thr in thresholds:
            correct1 = 0.0
            correct5 = 0.0
            total = 0
            frames_used_sum = 0.0
            conf_sum = 0.0
            latencies = []

            for i, (clip, label) in enumerate(loader):
                clip = clip.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                B, C, T, H, W = clip.shape
                k_eff = min(int(k), T)

                clip_sel, _ = select_topk_frames(clip, k=k_eff, score_type=score_type)

                do_measure = (i >= num_warmup) and (i < num_warmup + num_measure)
                if do_measure and device.type == "cuda":
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()

                final_logits, stats = streaming_early_exit(
                    backbone=model.backbone,
                    classifier=model.classifier,
                    clip=clip_sel,
                    threshold=float(thr),
                    min_frames=min_frames,
                    max_frames=k_eff,
                    frame_step=frame_step,
                )

                if do_measure and device.type == "cuda":
                    ender.record()
                    torch.cuda.synchronize()
                    latencies.append(starter.elapsed_time(ender))

                acc = accuracy_topk(final_logits, label, topk=topk_eval)
                correct1 += acc[1] * B
                correct5 += acc[5] * B
                total += B

                frames_used_sum += stats.used_frames.float().sum().item()
                conf_sum += stats.final_conf.sum().item()

            top1 = correct1 / total
            top5 = correct5 / total
            avg_used = frames_used_sum / total
            avg_conf = conf_sum / total
            avg_latency = (sum(latencies) / len(latencies)) if len(latencies) else 0.0

            results_lines.append(
                f"{k_eff},{thr:.2f},{top1:.6f},{top5:.6f},{avg_used:.3f},{avg_conf:.4f},{avg_latency:.3f}\n"
            )

            msg = f"[INFO] Hybrid k={k_eff} thr={thr:.2f} Top1={top1:.4f} Top5={top5:.4f} avg_used={avg_used:.2f} lat_ms={avg_latency:.2f}"
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

    if cfg["output"]["save_csv"]:
        csv_path.write_text("".join(results_lines), encoding="utf-8")
        print(f"[INFO] Saved hybrid summary CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run dynamic inference experiments (early-exit / frame-gating / hybrid).")
    parser.add_argument("--base", type=str, default="configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--cfg", type=str, default="configs/dynamic.yaml", help="Path to dynamic.yaml")
    parser.add_argument("--mode", type=str, default=None, help="Override dynamic.mode (early_exit|frame_gating|hybrid)")
    parser.add_argument("--save_dir", type=str, default=None, help="Override output.save_dir")
    args = parser.parse_args()

    base_cfg = load_config(args.base)
    dyn_cfg = load_config(args.cfg)

    if args.mode is not None:
        dyn_cfg.setdefault("dynamic", {})
        dyn_cfg["dynamic"]["mode"] = args.mode
    if args.save_dir is not None:
        dyn_cfg.setdefault("output", {})
        dyn_cfg["output"]["save_dir"] = args.save_dir

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if base_cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dynamic.log"

    split_root = Path(base_cfg["paths"]["split_root"])
    split_file = split_root / dyn_cfg["dataset"]["split"]

    ds = LazyFrameDataset(
        split_file=str(split_file),
        mode="supervised",
        clip_len=base_cfg["dataset"]["clip_len"],
        stride=base_cfg["dataset"]["stride"],
        image_size=base_cfg["dataset"]["image_size"],
        seed=base_cfg["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=int(dyn_cfg["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=base_cfg["device"]["num_workers"],
        pin_memory=base_cfg["device"]["pin_memory"],
    )

    model = VideoClassifier(
        num_classes=int(dyn_cfg["dataset"]["num_classes"]),
        embed_dim=int(dyn_cfg["model"]["embed_dim"]),
    )
    load_finetune_ckpt(model, dyn_cfg["model"]["finetune_ckpt"], device)

    mode = dyn_cfg["dynamic"]["mode"]
    Path(dyn_cfg["output"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_f:
        msg = f"[INFO] Dynamic inference started, mode={mode}"
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

        if mode == "early_exit":
            run_early_exit(model, loader, device, dyn_cfg, log_f)
        elif mode == "frame_gating":
            run_frame_gating(model, loader, device, dyn_cfg, log_f)
        elif mode == "hybrid":
            run_hybrid(model, loader, device, dyn_cfg, log_f)
        else:
            raise RuntimeError(f"[ERROR] Unknown dynamic.mode: {mode}")

    print("[INFO] Dynamic inference finished")


if __name__ == "__main__":
    main()
