import sys
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, set_seed, ensure_dir, write_csv
from datasets.federated_split import make_class_shard_splits
from federated.fed_loop import run_fedavg
from federated.client_sim import client_update
from federated.utils_fed import VideoClassifier, build_loader
from federated.comm_cost import estimate_comm_mb_per_round, bytes_to_mb


@torch.no_grad()
def evaluate_topk(model, loader, device, topk=(1, 5)):
    model.eval()
    total = 0
    correct = {k: 0 for k in topk}

    for clip, y in loader:
        clip = clip.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(clip)
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)

        for k in topk:
            ok = pred[:, :k].eq(y.view(-1, 1)).any(dim=1).float().sum().item()
            correct[k] += ok
        total += y.size(0)

    return correct[1] / max(1, total), correct[5] / max(1, total)


def _extract_state_dict(ckpt_obj):
    """
    Make ckpt loading robust across formats.
    """
    if ckpt_obj is None:
        return None
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # sometimes it is already a state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def load_init_ckpt_into_video_classifier(model: VideoClassifier, ckpt_path: str):
    """
    Load finetune checkpoint into VideoClassifier(backbone+classifier) as much as possible.
    Supports:
      - full state_dict with keys 'backbone.*' and 'classifier.*'
      - backbone-only state_dict (then loads into model.backbone)
    """
    if ckpt_path is None:
        print("[INFO] init_ckpt not provided, using random init")
        return
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[INFO] init_ckpt not found: {ckpt_path}, using random init")
        return

    raw = torch.load(str(p), map_location="cpu")
    state = _extract_state_dict(raw)
    if state is None:
        print(f"[WARN] Unrecognized ckpt format: {ckpt_path}, using random init")
        return

    # Try full load first
    missing, unexpected = model.load_state_dict(state, strict=False)

    # If it looks like backbone-only, try loading into backbone
    # (heuristic: no key starts with 'backbone.' but shapes match backbone)
    if len(unexpected) > 0 and not any(k.startswith("backbone.") for k in state.keys()):
        try:
            mb, _ = model.backbone.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded backbone-only ckpt into model.backbone: {ckpt_path}")
            if mb:
                print(f"[INFO] Backbone missing keys: {len(mb)}")
        except Exception:
            pass

    print(f"[INFO] Loaded init_ckpt: {ckpt_path}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")


def train_centralized(cfg_base, cfg_fed, device, train_split, val_loader, out_dir):
    """
    Centralized baseline under the same total training budget as FL:
      epochs = rounds * local_epochs (unless overridden)
    """
    c_cfg = cfg_fed.get("centralized", {})
    if not bool(c_cfg.get("enabled", True)):
        return None

    rounds = int(cfg_fed["federated"]["rounds"])
    local_epochs = int(cfg_fed["federated"]["local_epochs"])
    epochs = c_cfg.get("epochs", None)
    if epochs is None:
        epochs = rounds * local_epochs
    epochs = int(epochs)

    bs = int(c_cfg.get("batch_size", cfg_fed["federated"]["batch_size"]))
    lr = float(c_cfg.get("lr", cfg_fed["federated"]["lr"]))
    wd = float(c_cfg.get("weight_decay", cfg_fed["federated"]["weight_decay"]))
    amp = bool(cfg_fed["runtime"]["amp"])

    # loader
    _, train_loader = build_loader(
        split_file=str(train_split),
        base_cfg=cfg_base,
        batch_size=bs,
        mode="supervised",
        seed=int(cfg_base["seed"]) + 123,
        shuffle=True,
    )

    # model
    num_classes = int(cfg_fed["dataset"]["num_classes"])
    embed_dim = int(cfg_fed["model"]["embed_dim"])
    model = VideoClassifier(num_classes=num_classes, embed_dim=embed_dim).to(device)
    load_init_ckpt_into_video_classifier(model, cfg_fed["model"].get("init_ckpt"))

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    rows = []
    for ep in range(1, epochs + 1):
        total_loss, total = 0.0, 0
        for clip, y in train_loader:
            clip = clip.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(clip)
                loss = ce(logits, y)

            if amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs0 = y.size(0)
            total_loss += loss.item() * bs0
            total += bs0

        avg_loss = total_loss / max(1, total)
        val_top1, val_top5 = evaluate_topk(model, val_loader, device, topk=(1, 5))
        print(f"[INFO][Centralized] ep={ep}/{epochs} train_loss={avg_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}")

        rows.append({
            "epoch": ep,
            "train_loss": round(avg_loss, 6),
            "val_top1": round(val_top1, 6),
            "val_top5": round(val_top5, 6),
        })

    out_csv = Path(out_dir) / "centralized_summary.csv"
    write_csv(out_csv, ["epoch", "train_loss", "val_top1", "val_top5"], rows)
    print(f"[INFO] Saved centralized summary: {out_csv}")
    return rows


def estimate_raw_upload_mb(cfg_base, cfg_fed, train_split):
    """
    Rough proxy: if centralized training requires uploading raw clips to server once.
    raw_bytes ~= N_samples * (C * T * H * W * dtype_bytes)
    C=3, dtype_bytes default=1 (uint8)
    """
    if not bool(cfg_fed.get("system_privacy", {}).get("estimate_raw_upload", True)):
        return None

    dtype_bytes = int(cfg_fed.get("system_privacy", {}).get("raw_dtype_bytes", 1))
    clip_len = int(cfg_base["dataset"]["clip_len"])
    img = int(cfg_base["dataset"]["image_size"])
    C = 3
    per_sample_bytes = C * clip_len * img * img * dtype_bytes

    # count samples in split file
    n = 0
    with open(train_split, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1

    total_bytes = n * per_sample_bytes
    return bytes_to_mb(total_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/federated.yaml")
    args = ap.parse_args()

    cfg_base = load_config("configs/base.yaml")
    cfg_fed = load_config(args.config)

    set_seed(cfg_base["seed"])

    device = torch.device("cuda" if (cfg_base["device"]["use_cuda"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    split_root = Path(cfg_base["paths"]["split_root"])
    train_split = split_root / cfg_fed["dataset"]["train_split"]
    val_split = split_root / cfg_fed["dataset"]["val_split"]

    out_dir = ensure_dir(cfg_fed["output"]["save_dir"])
    log_dir = ensure_dir(cfg_base["paths"]["log_dir"])
    log_path = Path(log_dir) / "federated.log"

    # global validation loader
    _, val_loader = build_loader(
        split_file=str(val_split),
        base_cfg=cfg_base,
        batch_size=int(cfg_fed["federated"]["batch_size"]),
        mode="supervised",
        seed=int(cfg_base["seed"]) + 999,
        shuffle=False,
    )

    # ===== Centralized baseline (same budget) =====
    centralized_rows = train_centralized(cfg_base, cfg_fed, device, train_split, val_loader, out_dir)

    # ===== Build non-IID client splits =====
    num_clients = int(cfg_fed["federated"]["num_clients"])
    out_prefix = cfg_fed["output"]["split_prefix"]

    if cfg_fed["federated"]["non_iid"]["enabled"]:
        client_split_paths, client_stats = make_class_shard_splits(
            base_split_file=str(train_split),
            num_clients=num_clients,
            shards_per_client=int(cfg_fed["federated"]["non_iid"]["shards_per_client"]),
            seed=int(cfg_base["seed"]),
            min_samples_per_client=int(cfg_fed["federated"]["non_iid"]["min_samples_per_client"]),
            out_prefix=out_prefix,
            out_dir=str(split_root),
        )
        stats_csv = Path(out_dir) / "fed_client_stats.csv"
        lines = ["client,num_samples,num_classes,classes\n"]
        for s in client_stats:
            lines.append(f"{s['client']},{s['num_samples']},{s['num_classes']},{s['classes']}\n")
        stats_csv.write_text("".join(lines), encoding="utf-8")
        print(f"[INFO] Saved client stats: {stats_csv}")
    else:
        raise RuntimeError("[ERROR] IID mode not implemented; enable non_iid.")

    # ===== Build global model & client models =====
    num_classes = int(cfg_fed["dataset"]["num_classes"])
    embed_dim = int(cfg_fed["model"]["embed_dim"])

    global_model = VideoClassifier(num_classes=num_classes, embed_dim=embed_dim).to(device)
    load_init_ckpt_into_video_classifier(global_model, cfg_fed["model"].get("init_ckpt"))

    client_models = [VideoClassifier(num_classes=num_classes, embed_dim=embed_dim).to(device) for _ in range(num_clients)]

    # ===== Build client loaders =====
    client_sizes = []
    client_loaders = []

    for cid in range(num_clients):
        ds, loader = build_loader(
            split_file=client_split_paths[cid],
            base_cfg=cfg_base,
            batch_size=int(cfg_fed["federated"]["batch_size"]),
            mode="supervised",
            seed=int(cfg_base["seed"]) + cid,
            shuffle=True,
        )
        client_sizes.append(len(ds))

        def make_update_fn(_loader):
            def _fn(_model):
                return client_update(
                    model=_model,
                    loader=_loader,
                    device=device,
                    epochs=int(cfg_fed["federated"]["local_epochs"]),
                    lr=float(cfg_fed["federated"]["lr"]),
                    weight_decay=float(cfg_fed["federated"]["weight_decay"]),
                    amp=bool(cfg_fed["runtime"]["amp"]),
                )
            return _fn

        client_loaders.append({"loader": loader, "update_fn": make_update_fn(loader)})

    # eval function
    def eval_fn(model):
        return evaluate_topk(model, val_loader, device, topk=(1, 5))

    # ===== Run FedAvg =====
    rounds = int(cfg_fed["federated"]["rounds"])
    frac = float(cfg_fed["federated"]["client_fraction"])

    with open(log_path, "a", encoding="utf-8") as log_f:
        records = run_fedavg(
            global_model=global_model,
            client_models=client_models,
            client_loaders=client_loaders,
            client_sizes=client_sizes,
            evaluate_fn=eval_fn,
            device=device,
            rounds=rounds,
            client_fraction=frac,
            amp=bool(cfg_fed["runtime"]["amp"]),
            log_f=log_f,
        )

    # save federated summary
    comm_total = 0.0
    rows = []
    for r in records:
        comm_total += float(r["comm_mb_round"])
        rows.append({
            "round": r["round"],
            "val_top1": round(float(r["val_top1"]), 6),
            "val_top5": round(float(r["val_top5"]), 6),
            "avg_local_loss": round(float(r["avg_local_loss"]), 6),
            "clients": int(r["clients"]),
            "model_mb": round(float(r["model_mb"]), 6),
            "comm_mb_round": round(float(r["comm_mb_round"]), 6),
            "comm_mb_total": round(float(comm_total), 6),
        })

    fed_csv = Path(out_dir) / "fed_summary.csv"
    write_csv(
        fed_csv,
        ["round", "val_top1", "val_top5", "avg_local_loss", "clients", "model_mb", "comm_mb_round", "comm_mb_total"],
        rows
    )
    print(f"[INFO] Saved federated summary: {fed_csv}")
    print(f"[INFO] Log file: {log_path}")

    # ===== System-level privacy proxy summary =====
    raw_upload_mb = estimate_raw_upload_mb(cfg_base, cfg_fed, str(train_split))
    fed_comm_mb = comm_total

    sys_rows = [{
        "raw_upload_mb_est": round(raw_upload_mb, 6) if raw_upload_mb is not None else "",
        "fed_comm_total_mb": round(fed_comm_mb, 6),
        "reduction_ratio": round((fed_comm_mb / raw_upload_mb), 6) if (raw_upload_mb is not None and raw_upload_mb > 0) else "",
    }]
    sys_csv = Path(out_dir) / "system_privacy_summary.csv"
    write_csv(sys_csv, ["raw_upload_mb_est", "fed_comm_total_mb", "reduction_ratio"], sys_rows)
    print(f"[INFO] Saved system privacy summary: {sys_csv}")


if __name__ == "__main__":
    main()
