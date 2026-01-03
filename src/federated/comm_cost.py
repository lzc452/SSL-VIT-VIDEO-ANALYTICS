import torch


def model_size_bytes(state_dict):
    total = 0
    for _, v in state_dict.items():
        if torch.is_tensor(v):
            vv = v.detach().cpu()
            total += vv.numel() * vv.element_size()
    return int(total)


def bytes_to_mb(x):
    return float(x) / (1024.0 * 1024.0)


def estimate_comm_mb_per_round(global_state, num_clients_participating):
    """
    Typical FedAvg communication per round:
    - Server -> clients broadcast: N * model_size
    - Clients -> server upload:   N * model_size
    Total = 2N * model_size
    """
    size_b = model_size_bytes(global_state)
    total_b = int(2 * int(num_clients_participating) * size_b)
    return bytes_to_mb(total_b), bytes_to_mb(size_b)
