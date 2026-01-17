from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def mae_token_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B, N, D]
    mask: [B, N] True=masked
    """
    # only masked
    pred_m = pred[mask]
    tgt_m = target[mask]
    if pred_m.numel() == 0:
        return torch.zeros((), device=pred.device)
    return F.mse_loss(pred_m, tgt_m)


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z: [B, N, D] -> flatten to [M, D]
    Returns:
      var_loss, cov_loss
    """
    x = z.reshape(-1, z.shape[-1])  # [M,D]
    x = x - x.mean(dim=0, keepdim=True)

    std = torch.sqrt(x.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(gamma - std))

    # covariance
    cov = (x.T @ x) / (x.shape[0] - 1 + 1e-6)  # [D,D]
    diag = torch.diagonal(cov)
    cov = cov - torch.diag(diag)
    cov_loss = (cov ** 2).sum() / x.shape[1]
    return var_loss, cov_loss


def total_loss(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    cfg_mae: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MAE token loss + optional VICReg stabilizer (on encoder tokens).
    """
    loss_main = mae_token_mse(pred_tokens, target_tokens, mask)

    stats = {"loss_mae": float(loss_main.item())}

    if not bool(cfg_mae.get("vicreg_enable", False)):
        return loss_main, stats

    lam = float(cfg_mae.get("vicreg_lambda", 1.0))
    mu = float(cfg_mae.get("vicreg_mu", 1.0))
    nu = float(cfg_mae.get("vicreg_nu", 0.04))
    gamma = float(cfg_mae.get("vicreg_var_gamma", 1.0))

    # Invariance term: cosine alignment on visible tokens (方向一致，避免 scale trick)
    vis = ~mask
    p = pred_tokens[vis]
    t = target_tokens[vis].detach()
    if p.numel() == 0:
        inv = torch.zeros((), device=pred_tokens.device)
    else:
        p = F.normalize(p, dim=-1)
        t = F.normalize(t, dim=-1)
        inv = 2.0 - 2.0 * (p * t).sum(dim=-1).mean()

    var_l, cov_l = vicreg_terms(pred_tokens, gamma=gamma)

    loss = lam * inv + mu * var_l + nu * cov_l + loss_main

    stats.update(
        {
            "loss_inv": float(inv.item()),
            "loss_var": float(var_l.item()),
            "loss_cov": float(cov_l.item()),
            "loss_total": float(loss.item()),
        }
    )
    return loss, stats
