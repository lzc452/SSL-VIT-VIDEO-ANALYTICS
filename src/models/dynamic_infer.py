import torch
import torch.nn.functional as F


def encode_frames_to_logits(backbone, head, clips):
    """
    将视频 clip 编码为逐帧分类 logits.

    输入:
      backbone: MobileViTS 实例
      head: ClassificationHead 实例
      clips: [B, C, T, H, W]

    输出:
      frame_logits: [B, T, num_classes]
    """
    B, C, T, H, W = clips.shape
    device = clips.device

    # [B, C, T, H, W] -> [B, T, C, H, W]
    clips = clips.permute(0, 2, 1, 3, 4).contiguous()
    frames = clips.view(B * T, C, H, W)  # [B*T, C, H, W]

    feat_map = backbone(frames)          # [B*T, D, h, w]
    feat_vec = feat_map.mean(dim=[2, 3]) # [B*T, D]

    # ClassificationHead 期望输入 [B, D, 1, 1]
    logits = head(feat_vec.unsqueeze(-1).unsqueeze(-1))  # [B*T, num_classes]

    frame_logits = logits.view(B, T, -1)  # [B, T, num_classes]
    return frame_logits


def temporal_dynamic_exit(logits_seq, threshold=0.7, min_frames=2):
    """
    在单个 clip 的帧序列上执行时序动态推理策略.

    输入:
      logits_seq: [T, num_classes] 逐帧 logits
      threshold: float, 早退置信度阈值
      min_frames: int, 至少使用的帧数

    输出:
      exit_index: int, 使用到的最后一帧索引 (0-based)
      pred_label: int, 早退时的预测类别
      conf: float, 早退时的最高置信度
    """
    T, C = logits_seq.shape
    probs_running = None
    pred_label = None
    conf = 0.0
    exit_index = T - 1

    for t in range(T):
        # [0..t] 的平均 logits -> softmax
        avg_logits = logits_seq[: t + 1].mean(dim=0, keepdim=True)  # [1, C]
        probs = F.softmax(avg_logits, dim=-1)                       # [1, C]
        max_conf, max_idx = probs.max(dim=-1)                       # [1], [1]

        max_conf_val = max_conf.item()
        max_idx_val = max_idx.item()

        # 记录当前状态
        probs_running = probs
        pred_label = max_idx_val
        conf = max_conf_val
        exit_index = t

        # 至少看到 min_frames 帧之后，才允许早退
        if (t + 1) >= min_frames and max_conf_val >= threshold:
            break

    return exit_index, pred_label, conf
