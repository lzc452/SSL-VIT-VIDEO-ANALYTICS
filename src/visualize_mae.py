import os
import torch
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as T
from PIL import Image

from models.tiny_vit import tiny_vit_21m_variant
from models.mae_vit_adapter import TinyVideoMAE
from datasets.mae_loader import LazyVideoMAEDataset, get_tube_mask
from utils import load_config

def unpatchify(x, p=8, h=14, w=14):
    """
    将预测的 patches 还原回像素视频帧
    x: [B, T*196, 192] -> [B, 3, T, 112, 112]
    """
    B, L, C = x.shape
    T = L // (h * w)
    # [B, T, h, w, p, p, 3]
    x = x.reshape(B, T, h, w, p, p, 3)
    # 维度转换回 [B, 3, T, H, W]
    x = torch.einsum('bthwpqc->bcthpw', x)
    return x.reshape(B, 3, T, h * p, w * p)

@torch.no_grad()
def visualize(checkpoint_path, video_path, output_path="visual_results"):
    cfg = load_config("configs/ssl_mae.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    encoder = tiny_vit_21m_variant(img_size=112, use_checkpoint=False).to(device)
    model = TinyVideoMAE(encoder, cfg).to(device)
    
    # 加载权重 (由于保存的是 encoder.state_dict，我们需要对应加载)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.encoder.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")

    # 2. 准备数据
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 模拟读取一个 clip
    frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])[:16]
    clip = []
    for f in frames:
        img = Image.open(os.path.join(video_path, f)).convert('RGB')
        clip.append(transform(img))
    
    # [1, 3, 16, 112, 112]
    input_tensor = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    B, C, T, H, W = input_tensor.shape

    # 3. 生成 Mask 并预测
    mask = get_tube_mask(B, T, 196, 0.9).to(device)
    pred = model(input_tensor, mask)

    # 4. 还原像素
    # 原始图像 (去归一化)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(device)
    orig = input_tensor * std + mean
    
    # 预测图像
    recon = unpatchify(pred)
    # 如果训练时用了 norm_pix_loss，这里需要额外的去归一化处理（此处简化）
    recon = recon * std + mean
    
    # 遮蔽图像 (将 mask 应用到原图上用于展示)
    mask_visual = mask.view(B, T, 14, 14).unsqueeze(2).repeat(1, 1, 8*8, 1).view(B, T, 14, 14, 8, 8)
    mask_visual = torch.einsum('bthwpq->bthpwq', mask_visual).reshape(B, 1, T, 112, 112)
    masked_orig = orig * (1 - mask_visual)

    # 5. 保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f"{output_path}/reconstruction.mp4", fourcc, 8, (112*3, 112))

    # 转换为 numpy 格式并拼接 [0, 1] -> [0, 255]
    orig_np = (orig[0].permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    masked_np = (masked_orig[0].permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    recon_np = (recon[0].permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

    for t in range(16):
        # 拼接：原始 | 遮蔽 | 重建
        combined = np.hstack([orig_np[t], masked_np[t], recon_np[t]])
        # RGB to BGR for OpenCV
        out_video.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out_video.release()
    print(f"[SUCCESS] Result saved to {output_path}/reconstruction.mp4")

if __name__ == "__main__":
    # 示例用法：指定你的权重路径和 UCF101 的一个视频文件夹
    visualize(
        checkpoint_path="results/tinymae_v1/encoder_ep10.pth", 
        video_path="data/ucf101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01"
    )