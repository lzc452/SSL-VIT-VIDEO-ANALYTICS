# 简单估算本轮通信量（MB）：
# (模型大小 MB) × 2 (下发+上传) × 客户端数

def count_model_params(backbone, head):
    total = 0
    for p in backbone.parameters():
        total += p.numel()
    for p in head.parameters():
        total += p.numel()
    return total


def estimate_round_comm_mb(num_params, num_clients, bytes_per_param=4):
    """
    估算单轮通信开销 (MB):
    模型下发一次 + 上传一次 = 2 * 模型大小
    """
    model_mb = num_params * bytes_per_param / 1e6
    comm_mb = model_mb * 2 * num_clients
    return comm_mb
