import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """
    简化版 MobileNetV2 block:
    - 1x1 expand
    - 3x3 depthwise
    - 1x1 project
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_dim, k=1, s=1, p=0))
        # depthwise
        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                k=3,
                s=stride,
                p=1,
            )
        )
        # project
        layers.append(
            ConvBNAct(
                hidden_dim,
                out_ch,
                k=1,
                s=1,
                p=0,
                act=False,
            )
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = x + out
        return out


class TransformerBlock(nn.Module):
    """
    标准 Transformer encoder block:
    - LayerNorm
    - Multi-Head Self-Attention
    - MLP with GELU
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # x: [B, N, C]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT 风格模块:
    - Conv 提取局部特征
    - unfold 成序列, 用 Transformer 做全局建模
    - fold 回空间, 与输入残差相加
    """
    def __init__(self, in_ch, dim, depth=2):
        super().__init__()

        # local representation
        self.local_1 = ConvBNAct(in_ch, in_ch, k=3, s=1, p=1)
        self.local_2 = ConvBNAct(in_ch, dim, k=1, s=1, p=0)

        blocks = []
        for _ in range(depth):
            blocks.append(TransformerBlock(dim=dim, num_heads=4, mlp_ratio=2.0))
        self.global_transformer = nn.Sequential(*blocks)

        self.proj = ConvBNAct(dim, in_ch, k=1, s=1, p=0)

    def unfolding(self, x):
        # x: [B, C, H, W] -> [B, N, C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        return x, H, W

    def folding(self, x, H, W):
        # x: [B, N, C] -> [B, C, H, W]
        B, N, C = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x

    def forward(self, x):
        y = self.local_1(x)
        y = self.local_2(y)  # [B, dim, H, W]

        seq, H, W = self.unfolding(y)  # [B, N, dim]
        seq = self.global_transformer(seq)
        y = self.folding(seq, H, W)  # [B, dim, H, W]

        y = self.proj(y)  # [B, in_ch, H, W]
        out = x + y
        return out


class MobileViTS(nn.Module):
    """
    简化版 MobileViT-S 风格 backbone。
    输入: [B, 3, H, W] (H=W=112)
    输出: [B, embed_dim, H/16, W/16]
    """
    def __init__(self, in_channels=3, embed_dim=640):
        super().__init__()

        self.stem = ConvBNAct(in_channels, 16, k=3, s=2, p=1)

        # stage 1
        self.layer1 = nn.Sequential(
            InvertedResidual(16, 32, stride=1, expand_ratio=2.0),
            InvertedResidual(32, 32, stride=1, expand_ratio=2.0),
        )

        # stage 2
        self.layer2_mv2 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=2.0),
            InvertedResidual(64, 64, stride=1, expand_ratio=2.0),
        )
        self.layer2_mvit = MobileViTBlock(in_ch=64, dim=144, depth=2)

        # stage 3
        self.layer3_mv2 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=2.0),
        )
        self.layer3_mvit = MobileViTBlock(in_ch=96, dim=192, depth=4)

        # stage 4
        self.layer4_mv2 = nn.Sequential(
            InvertedResidual(96, 128, stride=2, expand_ratio=2.0),
        )
        self.layer4_mvit = MobileViTBlock(in_ch=128, dim=240, depth=3)

        self.conv_out = ConvBNAct(128, embed_dim, k=1, s=1, p=0)

        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x = self.layer2_mv2(x)
        x = self.layer2_mvit(x)

        x = self.layer3_mv2(x)
        x = self.layer3_mvit(x)

        x = self.layer4_mv2(x)
        x = self.layer4_mvit(x)

        x = self.conv_out(x)
        return x
