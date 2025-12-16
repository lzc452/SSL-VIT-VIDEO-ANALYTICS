import os
import sys
import time
from pathlib import Path

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

    @torch.no_grad()
    def forward_prefix(self, clip, t_used):
        """
        Early-exit evaluation: only use first t_used frames.
        clip: [B, C, T, H, W], t_used <= T
        """
        B, C, T, H, W = clip.shape
        t_used = min(t_used, T)

        feats = []
        for t in range(t_used):
            _, emb = self.backbone(clip[:, :, t, :, :])
            feats.append(emb)
        feats = torch.stack(feats, dim=1)   # [B, t_used, D]
        video_emb = feats.mean(dim=1)
        logits = self.classifier(video_emb)
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


def estimate_motion_scores(clip):
    """
    clip: [B,C,T,H,W] normalized tensor
    return scores: [B,T] higher means more motion/change
    Simple and cheap: L1 diff between consecutive frames (channel-wise)
    """
    # convert to per-frame magnitude
    # diff[t] = mean(|x_t - x_{t-1}|)
    B, C, T, H, W = clip.shape
    scores = torch.zeros((B, T), device=clip.device)
    if T <= 1:
        return scores

    diffs = (clip[:, :, 1:, :, :] - clip[:, :, :-1, :, :]).abs().mean(dim=(1, 3, 4))  # [B, T-1]
    scores[:, 1:] = diffs
    return scores


@torch.no_grad()
def run_early_exit(model, loader, device, cfg, log_f):
    thresholds = cfg["dynamic"]["confidence_thresholds"]
    min_frames = int(cfg["dynamic"]["min_frames"])
    max_frames = int(cfg["dynamic"]["max_frames"])
    step = int(cfg["dynamic"]["frame_step"])

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "early_exit_results.csv"

    topk = (1, 5)  # fixed for chapter 4 usage; you can make it yaml if needed

    results_lines = ["threshold,top1,top5,avg_frames,avg_conf,avg_latency_ms,throughput_fps\n"]

    # runtime settings
    num_warmup = int(cfg["runtime"]["num_warmup"])
    num_measure = int(cfg["runtime"]["num_measure"])
    use_amp = bool(cfg["runtime"]["amp"])

    for thr in thresholds:
        correct1 = 0.0
        correct5 = 0.0
        total = 0
        frames_used_sum = 0.0
        conf_sum = 0.0

        # latency measurement
        latencies = []

        for i, (clip, label) in enumerate(loader):
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B, C, T, H, W = clip.shape
            T = min(T, max_frames)

            # warmup / measure control (measure only first num_measure batches after warmup)
            do_measure = (i >= num_warmup) and (i < num_warmup + num_measure)

            if do_measure and device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()

            # Early exit loop (per batch)
            decided = torch.zeros((B,), dtype=torch.bool, device=device)
            used = torch.zeros((B,), dtype=torch.long, device=device)  # frames used
            final_logits = torch.zeros((B, cfg["dataset"]["num_classes"]), device=device)

            for t_used in range(min_frames, T + 1, step):
                logits = model.forward_prefix(clip[:, :, :T, :, :], t_used=t_used)
                prob = F.softmax(logits, dim=1)
                conf, _ = prob.max(dim=1)

                # decide newly satisfied samples
                new_decide = (~decided) & (conf >= thr)
                if new_decide.any():
                    final_logits[new_decide] = logits[new_decide]
                    used[new_decide] = t_used
                    decided[new_decide] = True

                # if all decided, stop
                if decided.all():
                    break

            # for remaining undecided, use max_frames or T
            if (~decided).any():
                t_final = T
                logits = model.forward_prefix(clip[:, :, :T, :, :], t_used=t_final)
                final_logits[~decided] = logits[~decided]
                used[~decided] = t_final
                decided[~decided] = True

            # measure end
            if do_measure and device.type == "cuda":
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))  # ms

            # accuracy
            acc = accuracy_topk(final_logits, label, topk=topk)
            correct1 += acc[1] * B
            correct5 += acc[5] * B
            total += B

            frames_used_sum += used.float().sum().item()

            # confidence summary (use final logits)
            conf = F.softmax(final_logits, dim=1).max(dim=1)[0]
            conf_sum += conf.sum().item()

        top1 = correct1 / total
        top5 = correct5 / total
        avg_frames = frames_used_sum / total
        avg_conf = conf_sum / total

        if len(latencies) > 0:
            avg_latency = sum(latencies) / len(latencies)
            # throughput in "frames per second" approx: (avg_frames per sample) / (time per sample)
            # we can output clips/s as well; here output FPS-equivalent for discussion
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

    # runtime settings
    num_warmup = int(cfg["runtime"]["num_warmup"])
    num_measure = int(cfg["runtime"]["num_measure"])
    use_amp = bool(cfg["runtime"]["amp"])

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

            # select frames (indices) per sample
            if score_type == "motion":
                scores = estimate_motion_scores(clip)  # [B,T]
                idx = scores.topk(k_eff, dim=1, largest=True).indices  # [B,k]
                idx, _ = idx.sort(dim=1)
            elif score_type == "random":
                idx = torch.stack([torch.randperm(T, device=device)[:k_eff].sort()[0] for _ in range(B)], dim=0)
            else:
                raise RuntimeError(f"[ERROR] Unknown gating_score: {score_type}")

            # gather frames
            # clip_sel: [B,C,k,H,W]
            idx_view = idx.view(B, 1, k_eff, 1, 1).expand(B, C, k_eff, H, W)
            clip_sel = torch.gather(clip, dim=2, index=idx_view)

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


def main():
    base_cfg = load_config("configs/base.yaml")
    dyn_cfg = load_config("configs/dynamic.yaml")

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if base_cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # logs
    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dynamic.log"

    # dataset (supervised mode)
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

    # model
    model = VideoClassifier(
        num_classes=int(dyn_cfg["dataset"]["num_classes"]),
        embed_dim=int(dyn_cfg["model"]["embed_dim"]),
    )
    load_finetune_ckpt(model, dyn_cfg["model"]["finetune_ckpt"], device)

    mode = dyn_cfg["dynamic"]["mode"]
    save_dir = Path(dyn_cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

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
            # hybrid: gating first, then early-exit within selected frames
            # For simplicity, run both summaries (paper can discuss combined curve)
            run_frame_gating(model, loader, device, dyn_cfg, log_f)
            run_early_exit(model, loader, device, dyn_cfg, log_f)
        else:
            raise RuntimeError(f"[ERROR] Unknown dynamic.mode: {mode}")

    print("[INFO] Dynamic inference finished")


if __name__ == "__main__":
    main()
