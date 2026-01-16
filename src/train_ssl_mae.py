import os
import time
import logging
from datetime import timedelta
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from models.tiny_vit import tiny_vit_21m_variant
from models.mae_vit_adapter import TinyVideoMAE
from datasets.mae_loader import LazyVideoMAEDataset, get_tube_mask
from utils import load_config, set_seed, save_checkpoint

# ==========================================
# 1. 辅助函数
# ==========================================

def format_time(seconds):
    """将秒数转换为可读的 HH:MM:SS 格式"""
    return str(timedelta(seconds=int(seconds)))

def patchify(imgs, p=8):
    B, C, T, H, W = imgs.shape
    hp, wp = H // p, W // p
    x = imgs.reshape(B, C, T, hp, p, wp, p)
    x = torch.einsum('bcth p w q -> b t h w p q c', x)
    return x.reshape(B, T * hp * wp, p * p * C)

def setup_logger(save_dir):
    exp_name = Path(save_dir).name
    log_dir = Path("logs") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_dir

# ==========================================
# 2. 增强版训练逻辑
# ==========================================

def train_one_epoch(model, loader, optimizer, scaler, epoch, device, config, writer, logger):
    model.train()
    ssl_cfg = config['ssl']
    train_cfg = config['training']
    
    num_steps = len(loader)
    batch_size = train_cfg['batch_size']
    
    # 计时统计初始化
    epoch_start_time = time.time()
    last_log_time = time.time()
    
    total_loss = 0
    
    for step, clip in enumerate(loader):
        clip = clip.to(device, non_blocking=True)
        B, C, T, H, W = clip.shape
        L = (H // 8) * (W // 8) 
        
        mask = get_tube_mask(B, T, L, ssl_cfg['mask_ratio']).to(device)
        target = patchify(clip, p=8)
        
        if ssl_cfg['norm_pix_loss']:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(clip, mask)
            loss_map = (pred - target) ** 2
            loss_map = loss_map.mean(dim=-1)
            mask_flatten = mask.flatten(1, 2)
            loss = (loss_map * mask_flatten).sum() / (mask_flatten.sum() + 1e-6)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # --- 增强日志显示逻辑 ---
        if step % train_cfg['log_interval'] == 0 and step > 0:
            current_time = time.time()
            # 计算这段间隔内的吞吐量 (Samples per second)
            elapsed_interval = current_time - last_log_time
            throughput = (train_cfg['log_interval'] * batch_size) / elapsed_interval
            
            # 计算本 Epoch 已消耗时间和预计剩余时间 (ETA)
            elapsed_epoch = current_time - epoch_start_time
            avg_step_time = elapsed_epoch / (step + 1)
            eta_seconds = avg_step_time * (num_steps - step - 1)
            
            pred_std = pred[mask_flatten.bool()].std().item()
            
            # 详细打印：包含 进度 | Loss | Std | 速度 | ETA
            logger.info(
                f"Epoch [{epoch}] [{step:4d}/{num_steps}] "
                f"Loss: {loss.item():.4f} | "
                f"Std: {pred_std:.3f} | "
                f"Speed: {throughput:.1f} samples/s | "
                f"Epoch ETA: {format_time(eta_seconds)}"
            )
            
            last_log_time = current_time
            
            # TensorBoard 记录
            global_step = (epoch - 1) * num_steps + step
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Throughput', throughput, global_step)

    return total_loss / num_steps, time.time() - epoch_start_time

# ==========================================
# 3. 主程序
# ==========================================

def main():
    cfg = load_config("configs/ssl_mae.yaml")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger, log_dir = setup_logger(cfg['training']['save_dir'])
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    encoder = tiny_vit_21m_variant(img_size=cfg['dataset']['image_size'], use_checkpoint=True)
    model = TinyVideoMAE(encoder, cfg).to(device)
    
    ds = LazyVideoMAEDataset(
        split_file=cfg['dataset']['train_split'],
        clip_len=cfg['dataset']['clip_len'],
        stride=cfg['dataset']['stride'],
        image_size=cfg['dataset']['image_size'],
        transform=transform
    )
    
    loader = DataLoader(
        ds, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True,
        prefetch_factor=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')
    
    logger.info("="*50)
    logger.info(f"STARTING PRETRAINING | Total Epochs: {cfg['training']['epochs']}")
    logger.info(f"image size: 112x112 | Mask Ratio: {cfg['ssl']['mask_ratio']} | clip_len: {['dataset']['clip_len']}  | datasets: {['dataset']['train_split']}")
    logger.info("="*50)
    
    total_start_time = time.time()

    for epoch in range(1, cfg['training']['epochs'] + 1):
        avg_loss, epoch_duration = train_one_epoch(model, loader, optimizer, scaler, epoch, device, cfg, writer, logger)
        
        # 记录全过程的总进度
        total_elapsed = time.time() - total_start_time
        avg_epoch_time = total_elapsed / epoch
        total_eta = avg_epoch_time * (cfg['training']['epochs'] - epoch)
        
        # Epoch 结束时的重磅总结
        logger.info("-" * 30)
        logger.info(f"==> Epoch {epoch} SUMMARY")
        logger.info(f"    Average Loss: {avg_loss:.4f}")
        logger.info(f"    Training Time: {format_time(epoch_duration)}")
        logger.info(f"    Cumulative Time: {format_time(total_elapsed)}")
        logger.info(f"    Total Training ETA: {format_time(total_eta)}")
        logger.info("-" * 30)
        
        if epoch % 10 == 0:
            ckpt_path = Path(cfg['training']['save_dir']) / f"encoder_ep{epoch}.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model.encoder.state_dict(), ckpt_path)
            logger.info(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()