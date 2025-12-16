import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, set_seed
from datasets.federated_split import make_class_shard_splits
from federated.fed_loop import run_fedavg
from federated.client_sim import client_update
from federated.utils_fed import VideoClassifier, try_load_ckpt, build_loader


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


def main():
    base_cfg = load_config("configs/base.yaml")
    fed_cfg = load_config("configs/federated.yaml")

    set_seed(base_cfg["seed"])

    device = torch.device("cuda" if (base_cfg["device"]["use_cuda"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # paths
    split_root = Path(base_cfg["paths"]["split_root"])
    train_split = split_root / fed_cfg["dataset"]["train_split"]
    val_split = split_root / fed_cfg["dataset"]["val_split"]

    out_dir = Path(fed_cfg["output"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(base_cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "federated.log"

    num_clients = int(fed_cfg["federated"]["num_clients"])
    out_prefix = fed_cfg["output"]["split_prefix"]

    # build non-IID client splits
    if fed_cfg["federated"]["non_iid"]["enabled"]:
        client_split_paths, client_stats = make_class_shard_splits(
            base_split_file=str(train_split),
            num_clients=num_clients,
            shards_per_client=int(fed_cfg["federated"]["non_iid"]["shards_per_client"]),
            seed=base_cfg["seed"],
            min_samples_per_client=int(fed_cfg["federated"]["non_iid"]["min_samples_per_client"]),
            out_prefix=out_prefix,
            out_dir=str(split_root),
        )
        # save stats
        stats_csv = out_dir / "fed_client_stats.csv"
        lines = ["client,num_samples,num_classes,classes\n"]
        for s in client_stats:
            lines.append(f"{s['client']},{s['num_samples']},{s['num_classes']},{s['classes']}\n")
        stats_csv.write_text("".join(lines), encoding="utf-8")
        print(f"[INFO] Saved client stats: {stats_csv}")
    else:
        raise RuntimeError("[ERROR] IID mode not implemented in this step (enable non_iid)")

    # global validation loader
    _, val_loader = build_loader(
        split_file=str(val_split),
        base_cfg=base_cfg,
        batch_size=int(fed_cfg["federated"]["batch_size"]),
        mode="supervised",
        seed=base_cfg["seed"] + 999,
        shuffle=False,
    )

    # build global model and clients
    num_classes = int(fed_cfg["dataset"]["num_classes"])
    embed_dim = int(fed_cfg["model"]["embed_dim"])

    global_model = VideoClassifier(num_classes=num_classes, embed_dim=embed_dim).to(device)
    try_load_ckpt(global_model, fed_cfg["model"].get("init_ckpt"))

    client_models = [VideoClassifier(num_classes=num_classes, embed_dim=embed_dim).to(device) for _ in range(num_clients)]

    # client loaders & update functions
    client_sizes = []
    client_loaders = []

    for cid in range(num_clients):
        ds, loader = build_loader(
            split_file=client_split_paths[cid],
            base_cfg=base_cfg,
            batch_size=int(fed_cfg["federated"]["batch_size"]),
            mode="supervised",
            seed=base_cfg["seed"] + cid,
            shuffle=True,
        )
        client_sizes.append(len(ds))

        def make_update_fn(_loader):
            def _fn(_model):
                return client_update(
                    model=_model,
                    loader=_loader,
                    device=device,
                    epochs=int(fed_cfg["federated"]["local_epochs"]),
                    lr=float(fed_cfg["federated"]["lr"]),
                    weight_decay=float(fed_cfg["federated"]["weight_decay"]),
                    amp=bool(fed_cfg["runtime"]["amp"]),
                )
            return _fn

        client_loaders.append({"loader": loader, "update_fn": make_update_fn(loader)})

    # evaluation function
    def eval_fn(model):
        return evaluate_topk(model, val_loader, device, topk=(1, 5))

    rounds = int(fed_cfg["federated"]["rounds"])
    frac = float(fed_cfg["federated"]["client_fraction"])

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
            amp=bool(fed_cfg["runtime"]["amp"]),
            log_f=log_f,
        )

    # save summary
    summary_csv = out_dir / "fed_summary.csv"
    lines = ["round,val_top1,val_top5,avg_local_loss,clients,model_mb,comm_mb_round\n"]
    for r in records:
        lines.append(
            f"{r['round']},{r['val_top1']:.6f},{r['val_top5']:.6f},{r['avg_local_loss']:.6f},"
            f"{r['clients']},{r['model_mb']:.3f},{r['comm_mb_round']:.3f}\n"
        )
    summary_csv.write_text("".join(lines), encoding="utf-8")
    print(f"[INFO] Saved federated summary: {summary_csv}")
    print(f"[INFO] Log file: {log_path}")


if __name__ == "__main__":
    main()
