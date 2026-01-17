from typing import Dict
import torch


@torch.no_grad()
def diag_stats(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    输出一些“塌陷/尺度/可见信息”诊断指标，避免你再被假收敛坑。
    """
    # global stats
    p = pred_tokens
    t = target_tokens

    out = {}
    out["pred_abs"] = float(p.abs().mean().item())
    out["pred_std"] = float(p.std(dim=0).mean().item())
    out["tgt_abs"] = float(t.abs().mean().item())
    out["tgt_std"] = float(t.std(dim=0).mean().item())

    vis = (~mask)
    if vis.any():
        out["pred_vis_abs"] = float(p[vis].abs().mean().item())
        out["tgt_vis_abs"] = float(t[vis].abs().mean().item())
    else:
        out["pred_vis_abs"] = 0.0
        out["tgt_vis_abs"] = 0.0

    if mask.any():
        out["pred_mask_abs"] = float(p[mask].abs().mean().item())
        out["tgt_mask_abs"] = float(t[mask].abs().mean().item())
    else:
        out["pred_mask_abs"] = 0.0
        out["tgt_mask_abs"] = 0.0

    out["mask_ratio_actual"] = float(mask.float().mean().item())
    return out
