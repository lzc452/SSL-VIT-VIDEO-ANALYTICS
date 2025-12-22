import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from datasets.transforms import build_transforms


# Dataset 行为
# 每次 __getitem__：
# 从该 video 的 frame 文件夹中
# Lazy 采样 T 帧（默认 16）
# 支持 stride
# resize → normalize
# 不提前加载全部帧
# 不生成 clip 文件
# SSL / supervised 两种模式

class LazyFrameDataset(Dataset):
    """
    Frame-Lazy Dataset:
    - Input: frame folder of a video
    - On-the-fly sampling frames to form a clip
    """

    def __init__(
        self,
        split_file,
        mode="ssl",
        clip_len=16,
        stride=2,
        image_size=112,
        seed=42,
    ):
        self.split_file = split_file
        self.mode = mode
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.seed = seed

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split()
                self.samples.append((Path(path), int(label)))

        if len(self.samples) == 0:
            raise RuntimeError("[ERROR] Empty split file")

        self.transforms = build_transforms(image_size)

        print(f"[INFO] Loaded {len(self.samples)} samples from {split_file}")
        print(f"[INFO] Dataset mode: {self.mode}")
        print(f"[INFO] Clip length: {self.clip_len}, Stride: {self.stride}")

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, num_frames, index):
        """
        Deterministic but diverse sampling:
        each sample has its own random offset based on index
        """
        rng = random.Random(self.seed + index)

        max_start = max(0, num_frames - self.clip_len * self.stride)
        start = rng.randint(0, max_start) if max_start > 0 else 0

        indices = [start + i * self.stride for i in range(self.clip_len)]
        return indices

    def _load_clip(self, frame_dir, index):
        frames = sorted(frame_dir.glob("*.jpg"))
        num_frames = len(frames)

        if num_frames == 0:
            print(f"[ERROR] No frames found in {frame_dir}")
            return None

        indices = self._sample_frame_indices(num_frames, index)

        clip = []
        for idx in indices:
            idx = min(idx, num_frames - 1)
            img_path = frames[idx]

            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"[ERROR] Failed to read image {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transforms(img)
            clip.append(img)

        clip = torch.stack(clip, dim=1)  # [C, T, H, W]
        return clip

    def __getitem__(self, index):
        frame_dir, label = self.samples[index]

        clip = self._load_clip(frame_dir, index)

        if clip == None:
            # 随机重采一个 index
            new_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_index)

        if self.mode == "ssl":
            return clip
        else:
            return clip, label
