import copy
import random
from typing import Dict, List

import torch

from federated.comm_cost import estimate_comm_mb_per_round


def _is_float_tensor(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and torch.is_floating_point(t)


def fedavg_aggregate(global_model, client_states: List[Dict[str, torch.Tensor]], client_weights: List[float]):
    """
    Dtype-safe FedAvg aggregation on state_dict.

    Key points (to avoid your Float->Long crash):
    - For floating-point tensors: weighted average
    - For non-floating tensors (e.g., BatchNorm num_batches_tracked, integer buffers):
        do NOT average. Keep a valid value (max for num_batches_tracked, else copy from first client).
    - Aggregation is performed on CPU to avoid GPU memory spikes.
    """
    if len(client_states) == 0:
        raise RuntimeError("[ERROR] No client states provided for aggregation.")
    if len(client_states) != len(client_weights):
        raise RuntimeError("[ERROR] client_states and client_weights length mismatch.")

    total_w = float(sum(client_weights))
    if total_w <= 0:
        raise RuntimeError("[ERROR] total client weight must be > 0.")

    global_state = global_model.state_dict()
    first_state = client_states[0]

    new_state = {}

    for k, g_t in global_state.items():
        # if any client missing key, fallback to global
        if any(k not in cs for cs in client_states):
            new_state[k] = g_t.detach().cpu()
            continue

        # floating -> weighted average
        if _is_float_tensor(g_t):
            acc = torch.zeros_like(g_t.detach().cpu())
            for cs, w in zip(client_states, client_weights):
                acc += cs[k].detach().cpu().to(acc.dtype) * (float(w) / total_w)
            new_state[k] = acc
        else:
            # non-float -> do not average
            if "num_batches_tracked" in k:
                # keep consistent counter: take max across clients
                vals = [cs[k].detach().cpu().to(torch.long) for cs in client_states]
                new_state[k] = torch.max(torch.stack(vals, dim=0), dim=0).values
            else:
                # copy from first client (or could keep global)
                new_state[k] = first_state[k].detach().cpu()

    # load back
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
    log_f=None,
):
    """
    Standard FedAvg training loop.

    Requirements:
    - client_loaders[cid]["update_fn"](model) -> returns avg_loss (float)
    - client_models are same architecture as global_model
    """
    num_clients = len(client_models)
    rng = random.Random(42)

    records = []

    for r in range(1, int(rounds) + 1):
        m = max(1, int(num_clients * float(client_fraction)))
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
            local_losses.append(float(loss))

            # IMPORTANT: detach + cpu to make aggregation stable and light
            st = {k: v.detach().cpu() for k, v in client_models[cid].state_dict().items()}
            client_states.append(st)
            client_weights.append(float(client_sizes[cid]))

        # aggregate (dtype-safe)
        new_state = fedavg_aggregate(global_model, client_states, client_weights)

        # comm cost (broadcast + upload)
        comm_total_mb, model_mb = estimate_comm_mb_per_round(new_state, num_clients_participating=len(selected))

        # evaluate global model
        val_acc1, val_acc5 = evaluate_fn(global_model)

        rec = {
            "round": r,
            "val_top1": float(val_acc1),
            "val_top5": float(val_acc5),
            "avg_local_loss": float(sum(local_losses) / max(1, len(local_losses))),
            "clients": int(len(selected)),
            "model_mb": float(model_mb),
            "comm_mb_round": float(comm_total_mb),
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
