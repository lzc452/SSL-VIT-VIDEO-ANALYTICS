import random
import cv2
import numpy as np
import torch
from pathlib import Path

from datasets.base_dataset import BaseVideoDataset
from datasets.transforms import build_transforms


class LazyVideoDataset(BaseVideoDataset):
    def __init__(
        self,
        split_file,
        mode="ssl",
        clip_len=16,
        stride=8,
        image_size=112,
        seed=42,
    ):
        super().__init__(split_file, mode)
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.rng = random.Random(seed)

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split()
                self.samples.append((path, int(label)))

        self.transforms = build_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, num_frames):
        max_start = max(0, num_frames - self.clip_len * self.stride)
        start = self.rng.randint(0, max_start) if max_start > 0 else 0
        return [start + i * self.stride for i in range(self.clip_len)]

    def _read_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[ERROR] Cannot open video: {video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(num_frames)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transforms(frame)
            frames.append(frame)

        cap.release()

        # pad if needed
        while len(frames) < self.clip_len:
            frames.append(frames[-1].clone())

        clip = torch.stack(frames, dim=1)  # [C,T,H,W]
        return clip

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        clip = self._read_clip(video_path)

        if self.mode == "ssl":
            return clip
        else:
            return clip, label
