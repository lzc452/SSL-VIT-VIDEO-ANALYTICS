import torch


# 本模块为 Federated Learning for System-Level Privacy提供：
# DP-SGD 风格梯度裁剪与噪声注入
# 模型更新扰动
# 可用于未来在 federated loop 中加入隐私机制

def clip_gradients(model, max_norm=1.0):
    """
    在 Federated Learning 中进行 gradient clipping，用于 Differential Privacy.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale)

    return total_norm


def add_noise_to_model(model, sigma=0.0):
    """
    在客户端更新后为模型参数加上噪声，以提升系统级隐私。
    """
    if sigma <= 0:
        return

    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * sigma
            p.add_(noise)


def dp_sgd_update(model, loss, optimizer, max_grad_norm=1.0, sigma=0.0):
    """
    在一个客户端执行 DP-SGD 更新:
      1) backward
      2) clip gradients
      3) add noise
      4) optimizer step
    """
    optimizer.zero_grad()
    loss.backward()

    clip_gradients(model, max_norm=max_grad_norm)
    add_noise_to_model(model, sigma=sigma)

    optimizer.step()
