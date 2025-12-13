import cv2
import numpy as np
import torch


# YuNet (OpenCV DNN) 人脸检测
# Gaussian blur高斯模糊
# Mosaic pixelation马赛克像素化
# Blackout停电
# 批量处理 [B,C,T,H,W]

def load_yunet(model_path="yunet.onnx"):
    """
    加载 YuNet 模型 (OpenCV DNN)。
    请确保 yunet.onnx 位于项目目录或提前下载。
    """
    model = cv2.FaceDetectorYN_create(
        model_path,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    return model


def gaussian_blur(face_region, ksize=25, sigma=8.0):
    return cv2.GaussianBlur(face_region, (ksize, ksize), sigma)


def mosaic(face_region, block_size=12):
    h, w = face_region.shape[:2]
    small = cv2.resize(face_region, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blackout(face_region):
    return np.zeros_like(face_region)


def apply_mask(frame, boxes, method, blur_ksize, blur_sigma, mosaic_size):
    """
    对单帧进行匿名化处理。
    """
    H, W, _ = frame.shape
    out = frame.copy()

    for box in boxes:
        x, y, w, h = box.astype(int)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        face = out[y1:y2, x1:x2]

        if method == "gaussian_blur":
            masked = gaussian_blur(face, ksize=blur_ksize, sigma=blur_sigma)
        elif method == "mosaic":
            masked = mosaic(face, block_size=mosaic_size)
        elif method == "blackout":
            masked = blackout(face)
        else:
            masked = face

        out[y1:y2, x1:x2] = masked

    return out


def anonymize_frames(clips, method="gaussian_blur",
                     blur_ksize=25, blur_sigma=8.0, mosaic_size=12):
    """
    输入:
        clips: [B, C, T, H, W]
    输出:
        anonymized_clips: [B, C, T, H, W]
    """
    device = clips.device
    B, C, T, H, W = clips.shape

    yunet = load_yunet()

    anon = torch.zeros_like(clips)

    for b in range(B):
        for t in range(T):
            frame = clips[b, :, t].permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            frame_uint8 = (frame * 255).astype(np.uint8)

            yunet.setInputSize((W, H))
            _, faces = yunet.detect(frame_uint8)

            boxes = []
            if faces is not None:
                for f in faces:
                    boxes.append(f[:4])

            masked = apply_mask(
                frame_uint8, boxes, method, blur_ksize, blur_sigma, mosaic_size
            )

            masked = masked.astype(np.float32) / 255.0
            masked_tensor = torch.from_numpy(masked).permute(2, 0, 1)  # [C,H,W]

            anon[b, :, t] = masked_tensor

    return anon.to(device)
