# src/models/tinyvit_backbone.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyViTBackbone(nn.Module):
    """
    TinyViT backbone wrapper (timm) that ALWAYS returns a fixed number of tokens:
      tokens: [B, pool_size*pool_size, C]

    Robustness rules:
    - If timm returns feature map [B,C,H,W] -> adaptive avg pool to (pool_size,pool_size) -> flatten.
    - If timm returns tokens [B,N,C]:
        * If N == pool_size^2 -> OK (already desired tokens).
        * Else if N-1 is a perfect square -> assume first token is class token, drop it, reshape to grid.
        * Else if N is a perfect square -> reshape directly to grid.
        * Else -> raise a clear error (NO silent mismatch).
      After reshape -> adaptive avg pool to (pool_size,pool_size) -> flatten.
    """

    def __init__(self, backbone: str = "tiny_vit_21m_224", pretrained: str = ""):
        super().__init__()
        import timm

        self.net = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")
        if pretrained:
            sd = torch.load(pretrained, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            # 不在这里 print，交给 train_mae.py 统一记录（避免多进程/多次初始化刷屏）
            self._load_info = {"missing": list(missing), "unexpected": list(unexpected)}
        else:
            self._load_info = {"missing": [], "unexpected": []}

        self._feat_dim: int | None = None

    @property
    def load_info(self):
        return self._load_info

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.net, "forward_features"):
            return self.net.forward_features(x)
        return self.net(x)

    @staticmethod
    def _is_perfect_square(n: int) -> bool:
        if n <= 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    @staticmethod
    def _to_grid_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] where N is a perfect square
        return:  [B, C, H, W]
        """
        B, N, C = tokens.shape
        s = int(math.isqrt(N))
        grid = tokens.transpose(1, 2).contiguous().view(B, C, s, s)
        return grid

    def forward(self, x: torch.Tensor, pool_size: int = 3) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return tokens: [B, pool_size*pool_size, C]
        """
        f = self.forward_features(x)

        # Case A: feature map [B,C,H,W]
        if f.dim() == 4:
            B, C, _, _ = f.shape
            self._feat_dim = C if self._feat_dim is None else self._feat_dim
            f = F.adaptive_avg_pool2d(f, output_size=(pool_size, pool_size))
            return f.flatten(2).transpose(1, 2).contiguous()  # [B,P,C]

        # Case B: tokens [B,N,C]
        if f.dim() == 3:
            B, N, C = f.shape
            self._feat_dim = C if self._feat_dim is None else self._feat_dim
            P = pool_size * pool_size

            # already desired token count
            if N == P:
                return f

            # try drop cls token
            if N > 1 and self._is_perfect_square(N - 1):
                tok = f[:, 1:, :]  # drop first token
                grid = self._to_grid_from_tokens(tok)  # [B,C,s,s]
                grid = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
                return grid.flatten(2).transpose(1, 2).contiguous()

            # try direct square reshape
            if self._is_perfect_square(N):
                grid = self._to_grid_from_tokens(f)
                grid = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
                return grid.flatten(2).transpose(1, 2).contiguous()

            raise RuntimeError(
                f"[TinyViTBackbone] forward_features returned tokens with N={N}, "
                f"cannot reshape to 2D grid. Expected N=={P} or square (or square+cls). "
                f"Please change backbone name or inspect timm model output."
            )

        raise RuntimeError(f"[TinyViTBackbone] Unsupported feature dims: {f.dim()}")
