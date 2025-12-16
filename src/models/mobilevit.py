import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Basic building blocks
# ---------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, exp=4, s=1):
        super().__init__()
        hidden = int(in_ch * exp)
        self.use_res = (s == 1 and in_ch == out_ch)
        self.pw1 = ConvBNAct(in_ch, hidden, k=1, s=1)
        self.dw = ConvBNAct(hidden, hidden, k=3, s=s, groups=hidden)
        self.pw2 = ConvBNAct(hidden, out_ch, k=1, s=1, act=False)

    def forward(self, x):
        y = self.pw2(self.dw(self.pw1(x)))
        return x + y if self.use_res else y


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    A simplified MobileViT block:
      - local conv (3x3) + pointwise
      - unfold to patches -> transformer
      - fold back -> fuse conv
    """
    def __init__(self, in_ch, out_ch, dim, patch_size=2, depth=2, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch_size = patch_size

        self.local1 = ConvBNAct(in_ch, in_ch, k=3, s=1)
        self.local2 = ConvBNAct(in_ch, dim, k=1, s=1)

        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads=heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        self.proj = ConvBNAct(dim, in_ch, k=1, s=1)
        self.fuse = ConvBNAct(in_ch + in_ch, out_ch, k=3, s=1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        y = self.local2(self.local1(x))  # [B, D, H, W]
        D = y.shape[1]

        ph = self.patch_size
        pw = self.patch_size

        # pad if needed
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = y.shape

        # unfold: [B, D, Hp, Wp] -> [B, N, D]
        y = y.reshape(B, D, Hp // ph, ph, Wp // pw, pw)
        y = y.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, gh, gw, ph, pw, D]
        y = y.view(B, (Hp // ph) * (Wp // pw) * (ph * pw), D)  # tokens

        # transformer
        y = self.transformer(y)  # [B, N, D]

        # fold back
        y = y.view(B, Hp // ph, Wp // pw, ph, pw, D)
        y = y.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, D, gh, ph, gw, pw]
        y = y.view(B, D, Hp, Wp)

        if pad_h or pad_w:
            y = y[:, :, :H, :W]

        y = self.proj(y)  # [B, C, H, W]
        z = torch.cat([x, y], dim=1)
        return self.fuse(z)


class MobileViTBackbone(nn.Module):
    """
    MobileViT-like lightweight CNN-Transformer hybrid backbone.
    Output: a spatial feature map and a global pooled embedding.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Stem
        self.stem = ConvBNAct(3, 16, k=3, s=2)

        # Stage 1
        self.s1 = nn.Sequential(
            InvertedResidual(16, 32, exp=4, s=2),
            InvertedResidual(32, 32, exp=4, s=1),
        )

        # Stage 2
        self.s2 = nn.Sequential(
            InvertedResidual(32, 64, exp=4, s=2),
            InvertedResidual(64, 64, exp=4, s=1),
        )

        # MobileViT blocks
        self.mvit1 = nn.Sequential(
            InvertedResidual(64, 64, exp=4, s=1),
            MobileViTBlock(64, 96, dim=128, patch_size=2, depth=2, heads=4),
        )
        self.mvit2 = nn.Sequential(
            InvertedResidual(96, 96, exp=4, s=2),
            MobileViTBlock(96, 128, dim=160, patch_size=2, depth=2, heads=4),
        )
        self.mvit3 = nn.Sequential(
            InvertedResidual(128, 128, exp=4, s=2),
            MobileViTBlock(128, 160, dim=192, patch_size=2, depth=2, heads=4),
        )

        self.head = nn.Sequential(
            ConvBNAct(160, embed_dim, k=1, s=1),
        )

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.mvit1(x)
        x = self.mvit2(x)
        x = self.mvit3(x)
        feat = self.head(x)  # [B, embed_dim, h, w]
        emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, embed_dim]
        return feat, emb


def build_mobilevit_s(embed_dim=256):
    return MobileViTBackbone(embed_dim=embed_dim)
