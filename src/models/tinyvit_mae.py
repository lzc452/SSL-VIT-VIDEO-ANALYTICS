from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tinyvit_backbone import TinyViTBackbone


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TinyViTMAE(nn.Module):
    """
    Feature-level MAE:
      - Encoder target = backbone stage4 tokens (no masking)
      - Student input = masked tokens (mask token)
      - Decoder reconstructs target tokens for masked positions
    """
    def __init__(
        self,
        image_size: int = 112,
        clip_len: int = 32,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        decoder_dim: int = 192,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        stage4_pool: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.clip_len = clip_len
        self.embed_dim = embed_dim
        self.stage4_pool = stage4_pool
        self.tokens_per_frame = stage4_pool * stage4_pool

        self.backbone = TinyViTBackbone(embed_dim=embed_dim, stage4_pool=stage4_pool)

        # token projection (backbone out already embed_dim, keep as identity but留接口)
        self.enc_proj = nn.Identity()

        # encoder positional embeddings for (T * tokens_per_frame)
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_len * self.tokens_per_frame, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # encoder transformer
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, encoder_mlp_ratio, drop, attn_drop)
            for _ in range(encoder_layers)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # decoder: map enc -> dec dim
        self.dec_embed = nn.Linear(embed_dim, decoder_dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, clip_len * self.tokens_per_frame, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, 4.0, drop, attn_drop)
            for _ in range(decoder_layers)
        ])
        self.dec_norm = nn.LayerNorm(decoder_dim)

        # predict back to token feature (embed_dim)
        self.dec_pred = nn.Linear(decoder_dim, embed_dim)

    @torch.no_grad()
    def encode_target(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: [B, C, T, H, W]
        return target tokens: [B, N, D], N=T*P*P
        """
        B, C, T, H, W = clip.shape
        x = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feat, _ = self.backbone(x)  # [B*T, D, P, P]
        P = feat.shape[-1]
        tokens = feat.flatten(2).transpose(1, 2)  # [B*T, P*P, D]
        tokens = self.enc_proj(tokens)
        tokens = tokens.reshape(B, T * (P * P), -1)
        return tokens

    def forward(self, clip: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip: [B,C,T,H,W]
        token_mask: [B,N] True=masked
        returns:
          pred_tokens: [B,N,D]
          target_tokens: [B,N,D] (detach outside if needed)
        """
        B, C, T, H, W = clip.shape
        assert T == self.clip_len, f"clip_len mismatch: got {T}, expect {self.clip_len}"

        # target from clean backbone tokens
        with torch.no_grad():
            target_tokens = self.encode_target(clip)  # [B,N,D]

        # student input tokens = target tokens (stopgrad copy) with mask token applied
        x = target_tokens.detach()

        # add pos
        pos = self.pos_embed[:, : x.shape[1], :]
        x = x + pos

        # apply mask token
        mt = self.mask_token.expand(B, x.shape[1], -1)
        x = torch.where(token_mask.unsqueeze(-1), mt, x)

        # encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # decoder
        y = self.dec_embed(x) + self.dec_pos[:, : x.shape[1], :]
        for blk in self.decoder:
            y = blk(y)
        y = self.dec_norm(y)

        pred_tokens = self.dec_pred(y)  # [B,N,D]
        return pred_tokens, target_tokens
