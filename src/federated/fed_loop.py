import copy
import random
import torch

from federated.comm_cost import estimate_comm_mb_per_round


def fedavg_aggregate(global_model, client_states, client_weights):
    """
    Weighted FedAvg on state_dict.
    """
    global_state = global_model.state_dict()
    new_state = {k: torch.zeros_like(v) for k, v in global_state.items()}

    total_w = sum(client_weights)
    for state, w in zip(client_states, client_weights):
        for k in new_state:
            new_state[k] += state[k] * (w / total_w)

    global_model.load_state_dict(new_state, strict=True)
    return new_state


def run_fedavg(
    global_model,
    client_models,
    client_loaders,
    client_sizes,
    evaluate_fn,
    device,
    rounds=10,
    client_fraction=1.0,
    amp=True,
    log_f=None
):
    num_clients = len(client_models)
    rng = random.Random(42)

    records = []
    for r in range(1, rounds + 1):
        m = max(1, int(num_clients * client_fraction))
        selected = rng.sample(list(range(num_clients)), m)

        msg = f"[INFO] Round {r}/{rounds} selected_clients={selected}"
        print(msg)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

        # broadcast global weights
        global_state = copy.deepcopy(global_model.state_dict())
        for cid in selected:
            client_models[cid].load_state_dict(global_state, strict=True)

        # local updates
        client_states = []
        client_weights = []

        local_losses = []
        for cid in selected:
            loss = client_loaders[cid]["update_fn"](client_models[cid])
            local_losses.append(loss)
            client_states.append(copy.deepcopy(client_models[cid].state_dict()))
            client_weights.append(client_sizes[cid])

        # aggregate
        new_state = fedavg_aggregate(global_model, client_states, client_weights)

        # comm cost
        comm_total_mb, model_mb = estimate_comm_mb_per_round(new_state, num_clients_participating=len(selected))

        # evaluate global model
        val_acc1, val_acc5 = evaluate_fn(global_model)

        rec = {
            "round": r,
            "val_top1": val_acc1,
            "val_top5": val_acc5,
            "avg_local_loss": sum(local_losses) / max(1, len(local_losses)),
            "clients": len(selected),
            "model_mb": model_mb,
            "comm_mb_round": comm_total_mb,
        }
        records.append(rec)

        msg = (
            f"[INFO] Round {r} val_top1={val_acc1:.4f} val_top5={val_acc5:.4f} "
            f"avg_local_loss={rec['avg_local_loss']:.4f} comm_mb={comm_total_mb:.2f}"
        )
        print(msg)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

    return records
