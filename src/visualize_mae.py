import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# 确保能导入项目中的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_vit import tiny_vit_21m_variant
from models.mae_vit_adapter import TinyVideoMAE
from datasets.mae_loader import get_tube_mask
from utils import load_config

def unpatchify(x, T=16, h=14, w=14, p=8):
    """标准还原逻辑：针对 602112 尺寸"""
    B = x.shape[0]
    x = x.reshape(B, T, h, w, p, p, 3)
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
    
    # 2. 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.encoder.load_state_dict(checkpoint)
    model.eval()
    print(f"[INFO] 权重加载成功: {checkpoint_path}")

    # 3. 图像读取 - 彻底绕过 numpy 转换
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])[:16]
    if len(frame_files) < 16:
        print(f"[ERROR] 帧数不足: {video_path}"); return

    clip = []
    m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for f in frame_files:
        # 直接使用 PIL 读取并转为原始字节流，完全不经过 np.array()
        img = Image.open(os.path.join(video_path, f)).convert('RGB').resize((112, 112))
        
        # 核心修复点：使用 ByteTensor 直接从 buffer 读取
        img_t = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img_t = img_t.view(112, 112, 3).permute(2, 0, 1).float().div(255.0).to(device)
        
        clip.append((img_t - m) / s)
    
    input_tensor = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # 4. 推理
    mask = get_tube_mask(1, 16, 196, 0.9).to(device)
    pred = model(input_tensor, mask)

    # 5. 后处理
    orig = input_tensor * s.view(3,1,1,1) + m.view(3,1,1,1)
    recon = unpatchify(pred) * s.view(3,1,1,1) + m.view(3,1,1,1)
    
    # 6. 遮罩生成 (避免 Bool 类型报错)
    mask_f = mask.float().view(16, 1, 14, 14)
    mask_px = F.interpolate(mask_f, size=(112, 112), mode='nearest')
    mask_px = mask_px.view(1, 16, 112, 112).unsqueeze(1).repeat(1, 3, 1, 1, 1)
    masked_orig = orig * (1 - mask_px)

    # 7. 保存视频 - 手动处理转 numpy 的过程，仅在最后一步
    save_path = os.path.join(output_path, "recon_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 8, (112*3, 112))

    for t in range(16):
        # 这里的转换是安全的，因为我们是在 Tensor 上调用 .numpy()，而不是试图从 numpy 转 tensor
        def to_cv2(t_img):
            im = t_img[0, :, t].permute(1, 2, 0).clamp(0, 1).cpu().detach()
            # 这里的转换不再依赖 numpy 的 C 接口对齐
            im_np = (im.numpy() * 255).astype('uint8')
            return cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        
        combined = np.hstack([to_cv2(orig), to_cv2(masked_orig), to_cv2(recon)])
        out.write(combined)

    out.release()
    print(f"[SUCCESS] 视频已生成: {save_path}")

if __name__ == "__main__":
    visualize("results/tinymae_v1/encoder_ep80.pth", "data/UCF101_frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01")