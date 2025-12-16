import torch


def model_size_bytes(state_dict):
    total = 0
    for _, v in state_dict.items():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def bytes_to_mb(x):
    return float(x) / (1024.0 * 1024.0)


def estimate_comm_mb_per_round(global_state, num_clients_participating):
    """
    FedAvg typical: server -> clients (broadcast) + clients -> server (upload)
    Total bytes ~ 2 * num_clients * model_size
    """
    size_b = model_size_bytes(global_state)
    total_b = 2 * num_clients_participating * size_b
    return bytes_to_mb(total_b), bytes_to_mb(size_b)
