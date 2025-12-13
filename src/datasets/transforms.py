import cv2
import numpy as np
import torch


# Resize → ToTensor → Normalize调整大小 → 转换为张量 → 归一化
# 使用 PyTorch + OpenCV（CV2）

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return img


class ToTensor:
    def __call__(self, img):
        # img: [H, W, C] → [C, H, W], 0-255 → float32
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


def build_transforms(image_size):
    mean = [0.485, 0.456, 0.406]  # standard ImageNet mean
    std = [0.229, 0.224, 0.225]   # standard ImageNet std

    return lambda img: Normalize(mean, std)(
        ToTensor()(
            Resize(image_size)(img)
        )
    )
