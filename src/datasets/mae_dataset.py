import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def _read_split(split_file: str) -> List[str]:
    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "data/UCF101_frames/.../v_xxx 0"
            parts = line.split()
            items.append(parts[0])
    return items


class MAEVideoFramesDataset(Dataset):
    """
    Lazy load frames from directory like:
      data/UCF101_frames/ClassName/video_id/0000.jpg
    Split line contains directory path WITHOUT image index.
    """
    def __init__(
        self,
        split_file: str,
        clip_len: int = 32,
        stride: int = 4,
        image_size: int = 112,
    ):
        self.video_dirs = _read_split(split_file)
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _list_frames(self, vdir: str) -> List[str]:
        p = Path(vdir)
        if not p.exists():
            # some splits might store relative path; try relative to repo root
            p = Path(".") / vdir
        files = sorted([str(x) for x in p.glob("*.jpg")])
        if not files:
            files = sorted([str(x) for x in p.glob("*.png")])
        return files

    def __getitem__(self, idx: int) -> torch.Tensor:
        vdir = self.video_dirs[idx]
        frames = self._list_frames(vdir)

        # If too short, loop pad
        need = (self.clip_len - 1) * self.stride + 1
        if len(frames) < need:
            if len(frames) == 0:
                # hard fail with clear error
                raise RuntimeError(f"No frames found in {vdir}")
            # repeat last frame
            last = frames[-1]
            while len(frames) < need:
                frames.append(last)

        # deterministic start = 0 for reproducibility (you can add random later)
        start = 0
        idxs = [start + i * self.stride for i in range(self.clip_len)]
        idxs = [min(i, len(frames) - 1) for i in idxs]

        clip = []
        for fi in idxs:
            img = Image.open(frames[fi]).convert("RGB")
            clip.append(self.tf(img))
        # [T,C,H,W] -> [C,T,H,W]
        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip
