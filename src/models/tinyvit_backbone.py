from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MBConv(nn.Module):
    """轻量稳定的替代块（工程上很稳，训练不容易炸）。"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 4):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.pw1 = ConvBNAct(in_ch, mid, 1, 1, 0, act=True)
        self.dw = ConvBNAct(mid, mid, 3, stride, 1, act=True)
        self.pw2 = ConvBNAct(mid, out_ch, 1, 1, 0, act=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.dw(self.pw1(x)))
        if self.use_res:
            y = y + x
        return self.act(y)


class TinyViTBackbone(nn.Module):
    """
    自包含 backbone，目标：
      - 输入 112x112
      - 走一个“TinyViT-style 分层下采样”的稳定版本
      - 最终用 AdaptiveAvgPool2d(pool, pool) 强制 Stage4 = pool x pool（你要的 3x3）
    注意：这份实现以“可跑 + 稳 + 方便 MAE + 易加 early-exit”为第一目标，
         不依赖外部 timm/官方 TinyViT 仓库，彻底自包含。
    """
    def __init__(self, embed_dim: int = 256, stage4_pool: int = 3):
        super().__init__()
        self.stage4_pool = stage4_pool

        # Stem: 112 -> 56
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2, 1, act=True),
            ConvBNAct(64, 64, 3, 1, 1, act=True),
        )

        # Stage 1: 56 -> 28
        self.stage1 = nn.Sequential(
            MBConv(64, 96, stride=2),
            MBConv(96, 96, stride=1),
        )

        # Stage 2: 28 -> 14
        self.stage2 = nn.Sequential(
            MBConv(96, 128, stride=2),
            MBConv(128, 128, stride=1),
            MBConv(128, 128, stride=1),
        )

        # Stage 3: 14 -> 7
        self.stage3 = nn.Sequential(
            MBConv(128, 192, stride=2),
            MBConv(192, 192, stride=1),
            MBConv(192, 192, stride=1),
        )

        # Stage 4: 7 -> 4 (再下采样一次会变 4x4)，然后强制池化到 3x3
        self.stage4 = nn.Sequential(
            MBConv(192, embed_dim, stride=2),
            MBConv(embed_dim, embed_dim, stride=1),
        )

        self.out_norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x: [B,3,H,W]
        return:
          feat: [B, D, P, P] where P=stage4_pool (default 3)
          (P,P)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.out_norm(x)

        # 强制对齐 3x3（你定稿要求）
        x = F.adaptive_avg_pool2d(x, (self.stage4_pool, self.stage4_pool))
        return x, (self.stage4_pool, self.stage4_pool)
