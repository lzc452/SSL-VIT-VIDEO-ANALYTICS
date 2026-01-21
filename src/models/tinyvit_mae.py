# src/models/tinyvit_mae.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .tinyvit_backbone import TinyViTBackbone


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int, decoder_dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.pred = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        x = self.pred(x)
        return x


class TinyViTMAE(nn.Module):
    """
    Input:  clip [B,C,T,H,W]
    Encode: TinyViT per-frame -> Stage4 tokens (s*s) with s=stage4_pool (default 3)
    Tokens: [B, T*s*s, D]
    Mask:   token_mask [B, N] (True=masked)
    Decode: Transformer decoder over full token sequence
    Loss computed on masked tokens only (handled outside)
    """
    def __init__(
        self,
        backbone_name: str,
        pretrained: str,
        stage4_pool: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.stage4_pool = int(stage4_pool)
        self.tokens_per_frame = self.stage4_pool * self.stage4_pool

        self.encoder = TinyViTBackbone(backbone=backbone_name, pretrained=pretrained)

        # embed_dim inferred lazily: run one forward in init is expensive; we initialize with a safe default
        # but we need a real embed dim to build decoder. We infer via a tiny dummy tensor once.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 112, 112)
            tok = self.encoder(dummy, pool_size=self.stage4_pool)  # [1, P, D]
            embed_dim = tok.shape[-1]

        self.embed_dim = int(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder = MAEDecoder(
            embed_dim=self.embed_dim,
            decoder_dim=int(decoder_dim),
            depth=int(decoder_depth),
            num_heads=int(decoder_num_heads),
            mlp_ratio=float(mlp_ratio),
        )

    def forward(self, clip: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip: [B,C,T,H,W]
        token_mask: [B,N] bool
        return:
          pred:   [N_mask, D]
          target: [N_mask, D]
        """
        B, C, T, H, W = clip.shape
        device = clip.device
        P = self.tokens_per_frame
        N = T * P

        assert token_mask.shape == (B, N), f"token_mask must be [B,{N}] got {tuple(token_mask.shape)}"

        # batch frames: [B,T,C,H,W] -> [B*T,C,H,W]
        x = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).contiguous()

        # encode tokens per frame: [B*T, P, D]
        tok = self.encoder(x, pool_size=self.stage4_pool)

        # [B,T,P,D] -> [B, T*P, D]
        tok = tok.reshape(B, T * P, self.embed_dim)

        # apply mask token
        mask_tok = self.mask_token.expand(B, T * P, self.embed_dim)
        tok_in = torch.where(token_mask.unsqueeze(-1), mask_tok, tok)

        # decode full sequence
        rec = self.decoder(tok_in)

        pred = rec[token_mask]
        target = tok[token_mask].detach()
        return pred, target
