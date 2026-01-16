import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
from typing import Tuple

# ==========================================
# 1. 基础组件 (适配 RTX 5090 与 BGR 修正)
# ==========================================

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio=4, stride=1, drop_path=0.):
        super().__init__()
        mid_chans = int(in_chans * expand_ratio)
        self.use_res_connect = stride == 1 and in_chans == out_chans
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2d_BN(in_chans, mid_chans, ks=1))
            layers.append(nn.GELU())
        layers.extend([
            Conv2d_BN(mid_chans, mid_chans, ks=3, stride=stride, pad=1, groups=mid_chans),
            nn.GELU(),
            SELayer(mid_chans),
            Conv2d_BN(mid_chans, out_chans, ks=1, bn_weight_init=0)
        ])
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        if self.use_res_connect:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)

# ==========================================
# 2. Transformer 组件 (Flash Attention 与 维度适配)
# ==========================================

class PatchEmbed(nn.Module):
    """针对 112x112 输入优化的 Stem，Stride 改为 1"""
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim // 2, ks=3, stride=2, pad=1),
            nn.GELU(),
            Conv2d_BN(embed_dim // 2, embed_dim, ks=3, stride=1, pad=1), # 修改点
        )
    def forward(self, x):
        return self.patch_embed(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Attention(nn.Module):
    """适配 RTX 5090 的 Flash Attention 版本"""
    def __init__(self, dim, key_dim, num_heads=8, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.qkv = nn.Linear(dim, key_dim * num_heads * 3)
        self.proj = nn.Linear(key_dim * num_heads, dim)

    def forward(self, x):
        B, L, C = x.shape
        # qkv: [B, L, 3, heads, key_dim] -> [3, B, heads, L, key_dim]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.key_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用高效的 Flash Attention 原生接口
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, L, -1)
        return self.proj(x)

class TinyViTBlock(nn.Module):
    """自动处理 B,C,H,W 与 B,L,C 转换，解决 LayerNorm 报错"""
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, key_dim=dim//num_heads, num_heads=num_heads, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 转换到 Transformer 序列格式
        shortcut = x
        x = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        # 2. 计算 (LayerNorm 现在能正确识别 dim)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 3. 还原到 CNN 格式适配下一层
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

# ==========================================
# 3. TinyViT 主类 (集成梯度检查点)
# ==========================================

class TinyViT(nn.Module):
    def __init__(self, img_size=112, in_chans=3, 
                 embed_dims=[96, 192, 384, 576], depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_sizes=[7, 7, 14, 7],
                 drop_path_rate=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(in_chans, embed_dims[0])
        
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(4):
            stage_blocks = []
            # Stage 下采样
            if i > 0:
                stage_blocks.append(MBConv(embed_dims[i-1], embed_dims[i], stride=2))
            
            # 内部 Block
            for j in range(depths[i]):
                if i == 0: # Stage 1 是全卷积
                    stage_blocks.append(MBConv(embed_dims[i], embed_dims[i], drop_path=dpr[cur]))
                else: # Stage 2-4 是 Transformer
                    stage_blocks.append(TinyViTBlock(
                        dim=embed_dims[i], num_heads=num_heads[i], 
                        window_size=window_sizes[i], drop_path=dpr[cur]))
                cur += 1
            self.stages.append(nn.Sequential(*stage_blocks))

    def forward_stage3(self, x):
        """MAE 预训练专用接口"""
        x = self.patch_embed(x)
        # 遍历前三个 Stage (Stage 1, 2, 3)
        for i in range(3):
            if self.use_checkpoint and self.training:
                # 使用梯度检查点显著降低显存
                x = checkpoint.checkpoint(self.stages[i], x, use_reentrant=False)
            else:
                x = self.stages[i](x)
        return x # 输出 [BT, 384, 14, 14]

    def forward(self, x):
        """分类/微调全流程接口"""
        x = self.patch_embed(x)
        for stage in self.stages:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x

def tiny_vit_21m_variant(img_size=112, use_checkpoint=True, **kwargs):
    return TinyViT(img_size=img_size, embed_dims=[96, 192, 384, 576], 
                   depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 18], 
                   use_checkpoint=use_checkpoint, **kwargs)