import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LazyVideoMAEDataset(Dataset):
    def __init__(self, split_file, clip_len=32, stride=2, image_size=112, transform=None):
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.transform = transform
        
        self.samples = []
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Index file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    self.samples.append(parts[0])

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, frame_path):
        try:
            return Image.open(frame_path).convert('RGB')
        except Exception:
            return Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

    def __getitem__(self, index):
        video_dir = self.samples[index]
        if not os.path.exists(video_dir):
            # 容错：返回全 0 张量
            return torch.zeros(3, self.clip_len, self.image_size, self.image_size)
            
        all_frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        total_frames = len(all_frames)
        
        if total_frames == 0:
            return torch.zeros(3, self.clip_len, self.image_size, self.image_size)

        # 【核心修复逻辑】：确保 indices 的长度永远等于 self.clip_len
        required_frames_window = self.clip_len * self.stride
        
        if total_frames < required_frames_window:
            # 情况 A：视频太短，不足以支撑所需的步长窗口
            # 直接通过 linspace 在现有帧中均匀抽取 clip_len 帧
            indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
        else:
            # 情况 B：视频足够长
            # 随机选择一个起始点，然后按照步长抽取 clip_len 帧
            start = np.random.randint(0, total_frames - required_frames_window + 1)
            indices = np.arange(start, start + required_frames_window, self.stride)
            
        # 确保万无一失：如果由于浮点精度导致多出一帧或少一帧，强制截断
        indices = indices[:self.clip_len]

        clip = []
        for i in indices:
            frame_path = os.path.join(video_dir, all_frames[i])
            img = self._load_frame(frame_path)
            
            if self.transform:
                img = self.transform(img)
            
            # 颜色修正逻辑 (BGR -> RGB)
            if isinstance(img, torch.Tensor):
                img = img[[2, 1, 0], :, :] 
                
            clip.append(img)
            
        # [T, C, H, W] -> [C, T, H, W]
        # 现在所有 Tensor 的 T 都是 32，stack 不会再报错
        clip = torch.stack(clip).permute(1, 0, 2, 3)
        return clip

def get_tube_mask(batch_size, num_frames, num_patches, mask_ratio):
    num_mask = int(mask_ratio * num_patches)
    masks = []
    for _ in range(batch_size):
        noise = torch.rand(num_patches)
        m = torch.zeros(num_patches)
        ids_mask = torch.argsort(noise, descending=True)[:num_mask]
        m[ids_mask] = 1
        masks.append(m)
    mask = torch.stack(masks)
    return mask.unsqueeze(1).repeat(1, num_frames, 1).bool()