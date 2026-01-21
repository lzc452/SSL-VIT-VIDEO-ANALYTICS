# src/datasets/mae_dataset.py
from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F


def _is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


class MAEVideoDataset(Dataset):
    """
    MAE dataset for frame-folder videos.
    Split file format: each line -> "<video_frame_dir> <label>"
    Label is ignored for SSL.

    Returns: clip tensor [C, T, H, W] float32 normalized (ImageNet mean/std).
    """

    def __init__(
        self,
        split_path: str,
        image_size: int = 112,
        clip_len: int = 32,
        stride: int = 4,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        training: bool = True,
        max_cache_videos: int = 20000,
    ):
        super().__init__()
        self.split_path = split_path
        self.image_size = int(image_size)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.training = bool(training)

        self.samples: List[str] = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # path label
                vdir = parts[0]
                self.samples.append(vdir)

        # LRU cache for frame lists
        self._list_frames = lru_cache(maxsize=max_cache_videos)(self._list_frames_uncached)

    def __len__(self) -> int:
        return len(self.samples)

    def _list_frames_uncached(self, vdir: str) -> Tuple[str, ...]:
        if not os.path.isdir(vdir):
            return tuple()
        try:
            names = [n for n in os.listdir(vdir) if _is_image_file(n)]
        except OSError:
            return tuple()
        names.sort()
        return tuple(os.path.join(vdir, n) for n in names)

    @staticmethod
    def _safe_read(path: str) -> Optional[torch.Tensor]:
        try:
            img = read_image(path, mode=ImageReadMode.RGB)  # uint8 [C,H,W]
            return img
        except Exception:
            return None

    def _resize_tensor(self, img_u8: torch.Tensor) -> torch.Tensor:
        """
        img_u8: uint8 [3,H,W] -> float32 [3,S,S]
        """
        x = img_u8.float().div_(255.0).unsqueeze(0)  # [1,3,H,W]
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x.squeeze(0).contiguous()

    def __getitem__(self, idx: int) -> torch.Tensor:
        vdir = self.samples[idx]
        frames = self._list_frames(vdir)
        n = len(frames)

        # If empty folder, return a safe zero clip (prevents DataLoader crash)
        if n == 0:
            clip = torch.zeros((3, self.clip_len, self.image_size, self.image_size), dtype=torch.float32)
            return (clip - self.mean) / self.std

        # choose start index
        max_start = max(0, n - (self.clip_len - 1) * self.stride - 1)
        if self.training:
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = 0

        # sample indices with clamp (short videos safe)
        indices = []
        for i in range(self.clip_len):
            j = start + i * self.stride
            if j >= n:
                j = n - 1
            indices.append(j)

        # clip-level augmentation: horizontal flip once per clip
        do_flip = self.training and (random.random() < 0.5)

        imgs: List[torch.Tensor] = []
        for j in indices:
            p = frames[j]
            img_u8 = self._safe_read(p)
            if img_u8 is None:
                # corrupted frame -> use zeros (but still keeps clip length correct)
                img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            else:
                img = self._resize_tensor(img_u8)  # float [3,S,S]
                if do_flip:
                    img = torch.flip(img, dims=[2])  # flip W dimension

            # normalize
            img = (img - self.mean) / self.std
            imgs.append(img)

        # stack to [T,3,S,S] -> [3,T,S,S]
        clip_t = torch.stack(imgs, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip_t
