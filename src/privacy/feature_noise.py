import torch


def add_gaussian_noise(z, sigma):
    if sigma <= 0:
        return z
    return z + torch.randn_like(z) * sigma


def apply_feature_mask(z, mask_ratio):
    if mask_ratio <= 0:
        return z
    keep_prob = 1.0 - mask_ratio
    mask = torch.bernoulli(torch.full_like(z, keep_prob))
    return z * mask
