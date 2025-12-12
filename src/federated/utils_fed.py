import os
from pathlib import Path
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.loader import VideoClipDataset
from models.backbone_mobilevit import MobileViTS
from models.heads import ClassificationHead


# 划分 Non-IID 客户端数据（根据 train_split 的 label）
# 构建全局模型（MobileViT-S + 分类头）
# 构建数据加载器、评估函数
# 提供 encode_clip

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_clip(backbone, clips):
    """
    将 clip 编码为 clip-level 特征.
    输入: clips [B, C, T, H, W]
    输出: clip_feat [B, D]
    """
    B, C, T, H, W = clips.shape
    device = clips.device

    clips = clips.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    frames = clips.view(B * T, C, H, W)                # [B*T, C, H, W]

    feat_map = backbone(frames)                        # [B*T, D, h, w]
    feat_vec = feat_map.mean(dim=[2, 3])               # [B*T, D]
    frame_feats = feat_vec.view(B, T, -1)              # [B, T, D]

    clip_feat = frame_feats.mean(dim=1)                # [B, D]
    return clip_feat


def create_federated_splits(global_split_path, num_clients, save_root, seed=42):
    """
    将全局 train_split 按 class-based Non-IID 划分到多个客户端:
      - 每个label只分配给少量客户端(这里是1个client)
      - 写出 client_k_train.txt 文件

    返回:
      client_split_paths: [path_1, ..., path_K]
      client_sample_counts: [n_1, ..., n_K]
    """
    fix_seed(seed)

    global_split_path = Path(global_split_path)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    with open(global_split_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 按 label 分组
    label_to_lines = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        label = int(parts[-1])
        label_to_lines.setdefault(label, []).append(line)

    labels = list(label_to_lines.keys())
    random.shuffle(labels)

    client_buckets = [[] for _ in range(num_clients)]

    # 类划分 Non-IID：每个 label 只给一个 client
    for idx, label in enumerate(labels):
        client_id = idx % num_clients
        client_buckets[client_id].extend(label_to_lines[label])

    client_split_paths = []
    client_sample_counts = []

    for cid in range(num_clients):
        client_lines = client_buckets[cid]
        client_sample_counts.append(len(client_lines))
        split_path = save_root / f"client_{cid+1}_train.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(client_lines))
        client_split_paths.append(str(split_path))

    return client_split_paths, client_sample_counts


def build_global_model(num_classes, init_checkpoint=""):
    """
    构建全局模型: MobileViT-S + ClassificationHead
    并可选加载初始化权重(例如 finetune_best.pth)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MobileViTS().to(device)
    head = ClassificationHead(backbone.embed_dim, num_classes).to(device)

    if init_checkpoint and Path(init_checkpoint).exists():
        print(f"[INFO] Loading init checkpoint: {init_checkpoint}")
        state = torch.load(init_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "backbone" in state and "head" in state:
            backbone.load_state_dict(state["backbone"], strict=False)
            head.load_state_dict(state["head"], strict=False)
        else:
            print("[ERROR] Init checkpoint format not as expected, skip loading.")
    else:
        if init_checkpoint:
            print(f"[ERROR] Init checkpoint not found: {init_checkpoint}, training from scratch.")

    return backbone, head


def build_val_loader(val_split, clip_len, image_size, batch_size, num_workers):
    dataset = VideoClipDataset(
        val_split,
        mode="supervised",
        clip_len=clip_len,
        image_size=image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def evaluate_global(backbone, head, val_loader, device, amp_enable=True):
    """
    评估当前全局模型的top-1准确率.
    """
    from torch.cuda.amp import autocast

    backbone.eval()
    head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=amp_enable):
                clip_feat = encode_clip(backbone, clips)  # [B, D]
                logits = head(clip_feat.unsqueeze(-1).unsqueeze(-1))

            _, preds = logits.max(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc
