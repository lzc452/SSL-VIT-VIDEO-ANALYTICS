import yaml
from pathlib import Path
import numpy as np

import torch

from federated.utils_fed import (
    fix_seed,
    create_federated_splits,
    build_global_model,
    build_val_loader,
    evaluate_global,
)
from federated.client_sim import ClientTrainer
from federated.comm_cost import count_model_params, estimate_round_comm_mb


# 实现 FedAvg：每轮
# 评估当前全局模型
# 挑选客户端（默认全参与）
# 每个客户端从全局权重出发做 local_train
# 按样本数加权平均 state_dict
# 记录 accuracy、round_comm_mb、cum_comm_mb、R*

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fedavg_aggregate(state_dicts, weights):
    """
    FedAvg 聚合:
    state_dicts: list of state_dict
    weights: list of float, sum=1
    """
    assert len(state_dicts) == len(weights)
    agg_state = {}

    for k in state_dicts[0].keys():
        agg_state[k] = 0.0
        for sd, w in zip(state_dicts, weights):
            agg_state[k] = agg_state[k] + sd[k] * w
    return agg_state


def run_federated_training(config_path):
    cfg = load_config(config_path)

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / cfg["output"]["summary_csv"]

    fix_seed(cfg["federated"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建 federated splits (Non-IID)
    global_train_split = cfg["dataset"]["train_split"]
    fed_split_root = Path("data/splits/federated")
    num_clients = cfg["federated"]["num_clients"]

    print("[INFO] Creating federated splits...")
    client_split_paths, client_sample_counts = create_federated_splits(
        global_train_split,
        num_clients=num_clients,
        save_root=fed_split_root,
        seed=cfg["federated"]["seed"],
    )

    for cid, (p, n) in enumerate(zip(client_split_paths, client_sample_counts), start=1):
        print(f"[INFO] Client {cid}: {n} samples, split_file={p}")

    # 初始化全局模型
    backbone, head = build_global_model(
        num_classes=cfg["dataset"]["num_classes"],
        init_checkpoint=cfg["model"]["init_checkpoint"],
    )

    num_params = count_model_params(backbone, head)
    print(f"[INFO] Total model parameters: {num_params}")

    # 验证集
    val_loader = build_val_loader(
        cfg["dataset"]["val_split"],
        clip_len=cfg["dataset"]["clip_len"],
        image_size=cfg["dataset"]["image_size"],
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
    )

    # 创建每个客户端的 trainer
    clients = []
    for cid, split_path in enumerate(client_split_paths, start=1):
        ct = ClientTrainer(
            client_id=cid,
            split_path=split_path,
            clip_len=cfg["dataset"]["clip_len"],
            image_size=cfg["dataset"]["image_size"],
            batch_size=cfg["dataloader"]["batch_size"],
            num_workers=cfg["dataloader"]["num_workers"],
            optimizer_cfg=cfg["optimizer"],
            local_epochs=cfg["federated"]["local_epochs"],
            amp_enable=True,
        )
        clients.append(ct)

    rounds = cfg["federated"]["rounds"]
    participation_rate = cfg["federated"]["participation_rate"]

    all_rounds = []
    cumulative_comm_mb = 0.0
    best_acc = 0.0
    best_r_star = 0.0

    print("[INFO] Start federated training...")

    for rnd in range(1, rounds + 1):
        # 全局评估
        acc = evaluate_global(backbone, head, val_loader, device, amp_enable=True)
        print(f"[INFO] Round {rnd}: global validation accuracy={acc:.4f}")

        # 客户端采样
        num_active = max(1, int(len(clients) * participation_rate))
        active_indices = np.random.choice(len(clients), size=num_active, replace=False)
        active_indices = sorted(active_indices.tolist())

        print(f"[INFO] Active clients in round {rnd}: {[i+1 for i in active_indices]}")

        # 备份当前全局权重
        global_backbone_state = {k: v.clone() for k, v in backbone.state_dict().items()}
        global_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        # 收集客户端更新
        client_backbone_states = []
        client_head_states = []
        client_weights = []

        for idx in active_indices:
            client = clients[idx]

            # 重置为同一全局权重
            backbone.load_state_dict(global_backbone_state, strict=True)
            head.load_state_dict(global_head_state, strict=True)

            sd_b, sd_h, n_samples = client.local_train(backbone, head, device)

            client_backbone_states.append(sd_b)
            client_head_states.append(sd_h)
            client_weights.append(n_samples)

        total_samples = float(sum(client_weights))
        client_weights = [w / total_samples for w in client_weights]

        # FedAvg 聚合
        new_backbone_state = fedavg_aggregate(client_backbone_states, client_weights)
        new_head_state = fedavg_aggregate(client_head_states, client_weights)

        backbone.load_state_dict(new_backbone_state, strict=True)
        head.load_state_dict(new_head_state, strict=True)

        # 通信开销
        round_comm_mb = estimate_round_comm_mb(num_params, num_active)
        cumulative_comm_mb += round_comm_mb

        # R*: accuracy / cumulative_comm_mb
        r_star = acc / cumulative_comm_mb if cumulative_comm_mb > 0 else 0.0

        if acc > best_acc:
            best_acc = acc
            best_r_star = r_star

        print(
            f"[INFO] Round {rnd}: acc={acc:.4f}, "
            f"round_comm_mb={round_comm_mb:.3f}, "
            f"cum_comm_mb={cumulative_comm_mb:.3f}, "
            f"R*={r_star:.6f}"
        )

        all_rounds.append(
            {
                "round": rnd,
                "acc": acc,
                "round_comm_mb": round_comm_mb,
                "cum_comm_mb": cumulative_comm_mb,
                "r_star": r_star,
            }
        )

    print(f"[INFO] Federated training finished. Best acc={best_acc:.4f}, best R*={best_r_star:.6f}")

    # 写 summary csv
    with open(summary_csv, "w") as f:
        f.write("round,acc,round_comm_mb,cum_comm_mb,r_star\n")
        for row in all_rounds:
            f.write(
                f"{row['round']},{row['acc']:.4f},"
                f"{row['round_comm_mb']:.4f},{row['cum_comm_mb']:.4f},"
                f"{row['r_star']:.6f}\n"
            )

    print(f"[INFO] Federated summary saved to {summary_csv}")
