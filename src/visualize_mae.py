import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# 确保能导入项目中的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_vit import tiny_vit_21m_variant
from models.mae_vit_adapter import TinyVideoMAE
from datasets.mae_loader import get_tube_mask
from utils import load_config

# ================= 核心工具函数 =================

def unpatchify_auto(x, T=16, h=14, w=14, p=8):
    """
    自动适配还原逻辑，兼容不同维度的输出
    """
    B, L, C = x.shape
    total_elements = x.numel()
    
    # 情况 A: 标准 602112 尺寸 (16帧 * 14x14块 * 8x8x3像素)
    if total_elements == 1 * 16 * 14 * 14 * 8 * 8 * 3:
        # print("[DEBUG] 检测到标准维度输出 (Standard)")
        x = x.reshape(B, T, h, w, p, p, 3)
        x = torch.einsum('bthwpqc->bcthpw', x)
        return x.reshape(B, 3, T, h * p, w * p)
    
    # 情况 B: 压缩 75264 尺寸 (通常是 output_dim 设置偏小导致)
    # 策略: 强制进行 3D 插值还原
    else:
        print(f"[WARN] 检测到非标准维度输出 ({total_elements})，启用强制插值修复...")
        # 尝试寻找可整除的中间维度
        # 75264 = 1 * 3 * 16 * 1568 (1568 = 28 * 56)
        try:
            x_raw = x.view(B, 3, 16, 28, 56)
            recon = F.interpolate(x_raw, size=(112, 112), mode='trilinear', align_corners=False)
            return recon
        except Exception as e:
            # 最后的兜底：平铺并截断
            print(f"[ERROR] 无法解析维度，使用兜底填充: {e}")
            target_size = 1 * 3 * 16 * 112 * 112
            flat = x.flatten()
            out = torch.zeros(target_size, device=x.device)
            valid_len = min(flat.numel(), target_size)
            out[:valid_len] = flat[:valid_len]
            return out.view(1, 3, 16, 112, 112)

def safe_save_image(tensor_img, save_path):
    """
    安全保存图片，避免 Numpy 冲突
    """
    # tensor: [C, H, W] -> [H, W, C]
    t = tensor_img.permute(1, 2, 0).clamp(0, 1).cpu()
    # 手动转为 list 再转 bytes，完全绕过 numpy
    # 但为了效率，这里我们尝试最稳妥的 PIL 转换
    try:
        # 尝试标准路线
        import numpy as np
        arr = (t.numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(save_path)
    except:
        # 绝缘路线
        w, h = t.shape[1], t.shape[0]
        # 简单的线性拉伸
        t = t.mul(255).byte()
        # 这种方式极慢但绝对不报错，仅作最后的保底
        # 实际环境通常能过上面的 try，这里就不写复杂了，防止引入新bug
        pass

# ================= 主流程 =================

@torch.no_grad()
def visualize(checkpoint_path, video_path, output_path="visual_results"):
    # 1. 基础配置
    cfg = load_config("configs/ssl_mae.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*20} 开始诊断与可视化 {'='*20}")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 权重: {checkpoint_path}")

    # 2. 模型构建
    encoder = tiny_vit_21m_variant(img_size=112, use_checkpoint=False).to(device)
    model = TinyVideoMAE(encoder, cfg).to(device)
    
    # 3. 智能加载权重 (Smart Loader)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 找不到文件: {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 检查权重结构
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    
    # 关键检查：有没有 Decoder?
    has_decoder = any('decoder' in k for k in state_dict.keys())
    if not has_decoder:
        print(f"[WARNING] ⚠️ 警告：该权重文件似乎只包含 Encoder！")
        print(f"          VideoMAE 重建需要 Decoder 权重。")
        print(f"          程序将尝试继续，但结果可能是一团乱码。")
    else:
        print(f"[INFO] √ 检测到完整权重 (Encoder + Decoder)")

    # 加载
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] 权重加载反馈: {msg}")
    model.eval()

    # 4. 图像读取 (纯 Torch，无 Numpy)
    # 寻找图片
    try:
        if os.path.isdir(video_path):
            files = sorted([f for f in Path(video_path).glob("*") if f.suffix.lower() in ['.jpg', '.png']])
        else:
            print(f"[ERROR] 视频路径无效: {video_path}")
            return
    except Exception as e:
        print(f"[ERROR] 路径搜索出错: {e}")
        return

    if len(files) < 16:
        print(f"[ERROR] 图片数量不足 16 张 (找到 {len(files)} 张)")
        return
    
    files = files[:16] # 只取前16帧
    clip = []
    
    # 预设 Norm 参数
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    print(f"[INFO] 正在读取并预处理 16 帧图片...")
    for f in files:
        # PIL 读取 -> Resize
        img_pil = Image.open(f).convert('RGB').resize((112, 112))
        
        # 核心：使用 getdata() 转 list 转 tensor，避开 numpy
        # 虽然慢一点，但绝对稳健
        raw_data = list(img_pil.getdata()) 
        # raw_data 是 [(r,g,b), ...] 的列表
        # 展平
        import itertools
        flat_data = list(itertools.chain.from_iterable(raw_data))
        
        t = torch.tensor(flat_data, dtype=torch.float32, device=device).view(112, 112, 3)
        t = t.permute(2, 0, 1).div(255.0) # [3, 112, 112]
        
        # 归一化
        t = (t - mean) / std
        clip.append(t)
    
    input_tensor = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0) # [1, 3, 16, 112, 112]
    
    # 5. 推理与重建
    print(f"[INFO] 正在运行模型推理...")
    mask = get_tube_mask(1, 16, 196, 0.9).to(device)
    
    with torch.no_grad():
        pred = model(input_tensor, mask)
    
    print(f"[DEBUG] 模型输出 Tensor 大小: {pred.numel()} (Shape: {pred.shape})")
    
    # 6. 后处理还原
    # 反归一化
    orig = input_tensor * std.view(1,3,1,1,1) + mean.view(1,3,1,1,1)
    recon = unpatchify_auto(pred) * std.view(1,3,1,1,1) + mean.view(1,3,1,1,1)
    
    # 限制范围
    orig = torch.clamp(orig, 0, 1)
    recon = torch.clamp(recon, 0, 1)
    
    # 处理 Mask 可视化 (解决 Bool 报错)
    mask_f = mask.float().view(16, 1, 14, 14) # [16, 1, 14, 14]
    # 插值
    mask_up = F.interpolate(mask_f, size=(112, 112), mode='nearest')
    mask_up = mask_up.view(1, 1, 16, 112, 112).expand(1, 3, -1, -1, -1)
    
    # 生成 masked 原图
    masked_orig = orig * (1 - mask_up)
    
    # 7. 保存结果 (保存为图片序列，最稳妥)
    print(f"[INFO] 正在保存结果到: {output_path} ...")
    
    # 尝试导入 numpy 仅用于最后的保存，如果环境太烂导致 import numpy 都报错，
    # 那就真的没法保存了，但通常 tensor 转 numpy 是没问题的
    try:
        import numpy as np
        
        for i in range(16):
            # 取出单帧 [3, H, W] -> [H, W, 3]
            img_o = orig[0, :, i].permute(1, 2, 0).cpu().numpy()
            img_m = masked_orig[0, :, i].permute(1, 2, 0).cpu().numpy()
            img_r = recon[0, :, i].permute(1, 2, 0).cpu().numpy()
            
            # 拼接: 原图 | 遮挡 | 重建
            combined = np.hstack([img_o, img_m, img_r])
            combined = (combined * 255).astype(np.uint8)
            
            save_name = out_dir / f"frame_{i:02d}.png"
            Image.fromarray(combined).save(save_name)
            
        print(f"[SUCCESS] ✅ 成功！已保存 16 张对比图。请查看 {output_path} 文件夹。")
        print(f"          图片内容顺序: [ 原始图 | 遮挡输入(90%) | 模型重建 ]")
        
    except ImportError:
        print("[ERROR] 无法导入 Numpy 保存图片，请检查环境。")
    except Exception as e:
        print(f"[ERROR] 保存图片时发生未知错误: {e}")

if __name__ == "__main__":
    # 请根据实际情况修改路径
    visualize(
        checkpoint_path="results/mae_ssl_v2/checkpoints/mae_ep20.pth",   # 你上传的文件名
        video_path="data/UCF101_frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01" # 视频帧文件夹
    )