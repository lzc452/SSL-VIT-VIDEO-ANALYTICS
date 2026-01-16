import torch
import torch.nn as nn
from timm.layers import trunc_normal_

class TinyVideoMAE(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        
        # 从配置中读取参数
        self.clip_len = config['dataset']['clip_len']    # 应该是 16
        self.img_size = config['dataset']['image_size']  # 应该是 112
        self.patch_size = 8
        self.num_patches = (self.img_size // self.patch_size) ** 2 # 14x14 = 196
        
        # 模型维度配置
        # TinyViT Stage 3 输出维度是 384
        self.encoder_dim = 384 
        self.decoder_embed_dim = config['model']['decoder_embed_dim']
        self.decoder_num_heads = config['model']['decoder_num_heads']
        self.decoder_depth = config['model']['decoder_depth']

        # 1. 维度转换层：将 Encoder 特征映射到 Decoder 维度
        self.enc_to_dec = nn.Linear(self.encoder_dim, self.decoder_embed_dim, bias=True)

        # 2. Mask Token：用于替换被遮盖部分的学习参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        # 3. 位置编码 (核心修改：对齐 16 帧)
        # 时间位置编码: [1, 16, 1, Dim]
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.clip_len, 1, self.decoder_embed_dim)
        )
        # 空间位置编码: [1, 1, 196, Dim]
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, self.decoder_embed_dim)
        )

        # 4. Decoder：使用轻量级 Transformer 块
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_embed_dim,
            nhead=self.decoder_num_heads,
            dim_feedforward=int(self.decoder_embed_dim * 4),
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=self.decoder_depth)
        
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        # 5. 预测头：映射回像素空间 (8*8*3 = 192)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size ** 2 * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # 位置编码初始化
        trunc_normal_(self.temporal_pos_embed, std=.02)
        trunc_normal_(self.spatial_pos_embed, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        # 线性层初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        """
        x: [B, 3, 16, 112, 112]
        mask: [B, 16, 196] (True 表示被遮盖)
        """
        B, C, T, H, W = x.shape
        
        # --- Step 1: Encoder 提取特征 ---
        # 将 T 帧合并到 Batch 处理: [B*16, 3, 112, 112]
        x_reshaped = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        
        # 提取 Stage 3 特征: [B*16, 384, 14, 14]
        latent = self.encoder.forward_stage3(x_reshaped)
        
        # 展平空间维度并转换维度: [B*16, 196, 384]
        latent = latent.flatten(2).transpose(1, 2)
        
        # 映射到 Decoder 维度并恢复 T 轴: [B, 16, 196, Dec_Dim]
        x_dec = self.enc_to_dec(latent).view(B, T, self.num_patches, -1)

        # --- Step 2: 注入位置信息 ---
        # 广播相加：[1, 16, 1, D] + [1, 1, 196, D] = [1, 16, 196, D]
        pos_embed = self.temporal_pos_embed + self.spatial_pos_embed
        x_dec = x_dec + pos_embed

        # --- Step 3: 应用 Mask 逻辑 ---
        # 将被遮盖位置的特征替换为 mask_token
        # mask 形状为 [B, 16, 196]，需要扩展维度以便广播
        mask_expanded = mask.unsqueeze(-1).type_as(x_dec)
        x_dec = x_dec * (1 - mask_expanded) + self.mask_token * mask_expanded

        # 展平为序列进入 Transformer Decoder: [B, 16 * 196, Dec_Dim]
        x_dec = x_dec.flatten(1, 2)

        # --- Step 4: Decoder 重建 ---
        x_dec = self.decoder_blocks(x_dec)
        x_dec = self.decoder_norm(x_dec)

        # --- Step 5: 像素预测 ---
        # [B, 3136, 192]
        pred = self.decoder_pred(x_dec)
        
        return pred