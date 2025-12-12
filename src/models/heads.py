import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    简单分类头:
    输入: [B, C, H, W]
    流程: GAP -> Linear(C -> num_classes)
    输出: logits [B, num_classes]
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feat_map):
        # feat_map: [B, C, H, W]
        x = feat_map.mean(dim=[2, 3])  # [B, C]
        logits = self.fc(x)
        return logits


class SSLMultiTaskHead(nn.Module):
    """
    自监督多任务 head:
    - Masked Feature Reconstruction (MFR): 重建被mask掉的frame-level特征
    - Temporal Order Prediction (TOP): 预测两帧的时间顺序（正常/反转）
    输入 feats: [B, T, D] (clip内每帧的特征向量)
    输出: loss 字典
    """
    def __init__(self, embed_dim, mask_ratio=0.7, enable_mfr=True, enable_top=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.enable_mfr = enable_mfr
        self.enable_top = enable_top

        # MFR: 简单 MLP autoencoder-style 重建
        self.mfr_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # TOP: 二分类，预测两个帧的时间顺序
        self.top_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feats):
        """
        feats: [B, T, D]
        返回:
          {
            "mfr": mfr_loss or 0,
            "top": top_loss or 0,
            "total": total_loss
          }
        """
        device = feats.device
        B, T, D = feats.shape

        total_loss = torch.tensor(0.0, device=device)
        mfr_loss = torch.tensor(0.0, device=device)
        top_loss = torch.tensor(0.0, device=device)

        if self.enable_mfr:
            # 随机mask一部分frames
            mask = torch.rand(B, T, device=device) < self.mask_ratio  # True 表示需要重建
            if mask.sum() == 0:
                mask[:] = True

            masked_feats = feats[mask]  # [N_mask, D]
            recon = self.mfr_head(masked_feats)
            mfr_loss = self.mse(recon, masked_feats)
            total_loss = total_loss + mfr_loss

        if self.enable_top and T >= 2:
            # 对每个clip随机取两帧，判断时间顺序
            idx_i = torch.randint(low=0, high=T, size=(B,), device=device)
            idx_j = torch.randint(low=0, high=T, size=(B,), device=device)
            same = idx_i == idx_j
            if same.any():
                idx_j[same] = (idx_j[same] + 1) % T

            feats_i = feats[torch.arange(B, device=device), idx_i]  # [B, D]
            feats_j = feats[torch.arange(B, device=device), idx_j]  # [B, D]

            pair_ij = torch.cat([feats_i, feats_j], dim=-1)  # label 1: i<j
            pair_ji = torch.cat([feats_j, feats_i], dim=-1)  # label 0: j<i

            pairs = torch.cat([pair_ij, pair_ji], dim=0)  # [2B, 2D]
            labels = torch.cat(
                [torch.ones(B, dtype=torch.long, device=device),
                 torch.zeros(B, dtype=torch.long, device=device)],
                dim=0
            )

            logits = self.top_head(pairs)  # [2B, 2]
            top_loss = self.ce(logits, labels)
            total_loss = total_loss + top_loss

        return {
            "mfr": mfr_loss,
            "top": top_loss,
            "total": total_loss,
        }
