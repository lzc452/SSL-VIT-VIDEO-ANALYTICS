import torch
import torch.nn.functional as F

# 添加 Gaussian noise添加高斯噪声
# 掩盖部分 feature map穿透部分特征图
# 计算 FLR（Feature Leakage Rate）计算 FLR（特征泄漏率）
# 计算 entropy

def add_gaussian_noise(feat_map, sigma):
    """
    feat_map: [B, C, h, w]
    sigma: float
    """
    if sigma <= 0:
        return feat_map

    noise = torch.randn_like(feat_map) * sigma
    return feat_map + noise


def apply_feature_mask(feat_map, mask_ratio):
    """
    随机mask部分空间区域
    feat_map: [B, C, h, w]
    mask_ratio: float 0~1
    """
    if mask_ratio <= 0:
        return feat_map

    B, C, H, W = feat_map.shape
    total_patches = H * W
    num_mask = int(total_patches * mask_ratio)

    feat_map = feat_map.clone()
    for b in range(B):
        idx = torch.randperm(total_patches, device=feat_map.device)[:num_mask]
        h_idx = idx // W
        w_idx = idx % W
        feat_map[b, :, h_idx, w_idx] = 0.0

    return feat_map


def compute_feature_leakage(feat_clean, feat_noisy):
    """
    计算 FLR (Feature Leakage Rate)
    使用 1 - (dist_noisy / dist_clean_ref) 的方式衡量隐私泄露程度.

    返回值越高表示泄露越严重。
    """
    B = feat_clean.shape[0]

    # L2 距离
    dist = torch.norm(feat_clean - feat_noisy, p=2, dim=[1, 2, 3])  # [B]
    ref = torch.norm(feat_clean, p=2, dim=[1, 2, 3])                # [B]

    flr = 1 - (dist / (ref + 1e-6))  # 保证不除零

    # 平均 FLR
    return flr.mean().item()


def compute_entropy(logits):
    """
    logits: [B, num_classes]
    返回平均 entropy
    """
    probs = F.softmax(logits, dim=-1)
    entropy = - torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
    return entropy.mean().item()
