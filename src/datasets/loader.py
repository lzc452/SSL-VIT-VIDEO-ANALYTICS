import os
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.transforms import build_transforms

# 加载 split 文件 <clip_path> <label>
# 返回 clip 的 tensor [C, T, H, W]
# 支持 mode="ssl" 与 mode="supervised"
# 支持 transforms支持转变
# 兼容 numpy .npy clip 文件（你 preprocess 会输出 .npy clips）

class VideoClipDataset(Dataset):
    """
    读取生成好的 clip 数组 (.npy)，每个 clip 是 [T, H, W, C]
    输出为 [C, T, H, W]，适配模型计算流程。

    模式:
      mode="ssl"  -> 返回 clips
      mode="supervised" -> 返回 (clips, label)
    """

    def __init__(self, split_file, mode="ssl", clip_len=16, image_size=112):
        super().__init__()
        self.split_file = Path(split_file)
        self.mode = mode
        self.clip_len = clip_len
        self.image_size = image_size

        if not self.split_file.exists():
            raise FileNotFoundError(f"[ERROR] split file not found: {self.split_file}")

        with open(self.split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.samples = []
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            clip_path = parts[0]
            label = int(parts[1])
            self.samples.append((clip_path, label))

        if len(self.samples) == 0:
            print(f"[ERROR] No valid samples found in split file: {self.split_file}")

        self.transforms = build_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, clip_path):
        """
        clip_path: path/to/xxx.npy
        .npy 内容为 [T, H, W, C]
        """
        arr = np.load(clip_path)  # [T, H, W, C]
        if arr.ndim != 4:
            raise ValueError(f"[ERROR] clip shape must be [T,H,W,C], but got {arr.shape}")

        T, H, W, C = arr.shape
        if T != self.clip_len:
            print(f"[INFO] Warning: Expected {self.clip_len} frames, found {T}. Auto-adjusting.")
            if T > self.clip_len:
                arr = arr[: self.clip_len]
            else:
                pad = np.repeat(arr[-1:], repeats=(self.clip_len - T), axis=0)
                arr = np.concatenate([arr, pad], axis=0)

        return arr

    def __getitem__(self, index):
        clip_path, label = self.samples[index]
        clip_np = self._load_clip(clip_path)

        frames = []
        for t in range(self.clip_len):
            img = clip_np[t]  # [H, W, C]
            img_tensor = self.transforms(img)  # [C, H, W]
            frames.append(img_tensor)

        clip_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]

        if self.mode == "ssl":
            return clip_tensor
        else:
            return clip_tensor, label
