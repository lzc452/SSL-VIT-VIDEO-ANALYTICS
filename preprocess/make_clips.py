import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import cv2

# -------------------------------------------
# Configurations
# -------------------------------------------
DATA_ROOT = "data"
CLIP_LEN = 16         # 每个 clip 含 16 帧
STRIDE = 8            # 每隔 8 帧采一个 clip
IMG_SIZE = (112, 112) # 图像大小 (H, W)
NUM_WORKERS = max(1, mp.cpu_count() // 2)

# -------------------------------------------
# Utility: load frame sequence and generate clips
# -------------------------------------------
def make_clips_from_video(args):
    frames_dir, save_root = args
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if len(frame_files) < CLIP_LEN:
            return f"[Skip] Too short: {frames_dir}"

        os.makedirs(save_root, exist_ok=True)
        total_clips = 0

        # 循环滑动窗口采样 clip
        for start in range(0, len(frame_files) - CLIP_LEN + 1, STRIDE):
            clip_name = f"clip_{start:05d}.npy"
            save_path = os.path.join(save_root, clip_name)
            if os.path.exists(save_path):
                continue  # 已存在跳过

            clip_frames = []
            for i in range(start, start + CLIP_LEN):
                frame_path = os.path.join(frames_dir, frame_files[i])
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames.append(frame)

            if len(clip_frames) == CLIP_LEN:
                np.save(save_path, np.array(clip_frames, dtype=np.uint8))
                total_clips += 1

        return f"[OK] {frames_dir} → {total_clips} clips"
    except Exception as e:
        return f"[Error] {frames_dir}: {e}"

# -------------------------------------------
# Process each dataset (e.g., UCF101_frames → UCF101_clips)
# -------------------------------------------
def process_dataset(frames_dataset):
    print(f"\n[Dataset] {frames_dataset}")
    input_root = os.path.join(DATA_ROOT, frames_dataset)
    clips_dataset = frames_dataset.replace("_frames", "_clips")
    output_root = os.path.join(DATA_ROOT, clips_dataset)

    print(f"\n[Dataset] {frames_dataset}")
    print(f" Input: {input_root}")
    print(f" Output: {output_root}\n")

    tasks = []

    # 遍历类别
    for cls in sorted(os.listdir(input_root)):
        cls_path = os.path.join(input_root, cls)
        if not os.path.isdir(cls_path):
            continue

        # 遍历视频帧目录
        for vid in sorted(os.listdir(cls_path)):
            vid_frames = os.path.join(cls_path, vid)
            if not os.path.isdir(vid_frames):
                continue

            save_root = os.path.join(output_root, cls, vid)
            if os.path.exists(save_root) and len(os.listdir(save_root)) > 0:
                continue  # 已存在跳过

            tasks.append((vid_frames, save_root))

    if not tasks:
        print(f"[Info] No new videos to process for {frames_dataset}.")
        return

    print(f"[Info] Found {len(tasks)} video folders → generating clips...")

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(make_clips_from_video, tasks),
                            total=len(tasks), ncols=80))

    ok = sum(1 for r in results if r.startswith("[OK]"))
    skip = sum(1 for r in results if r.startswith("[Skip]"))
    err = sum(1 for r in results if r.startswith("[Error]"))
    print(f"\n[Sucess] Done {frames_dataset}: {ok} OK, {skip} skipped, {err} errors.\n")

# -------------------------------------------
# Main: auto-detect *_frames datasets
# -------------------------------------------
if __name__ == "__main__":
    print("==========================================")
    print(" Generating Fixed-Length Clips from Frames ")
    print("==========================================")

    datasets = [d for d in os.listdir(DATA_ROOT)
                if d.endswith("_frames") and os.path.isdir(os.path.join(DATA_ROOT, d))]

    if not datasets:
        print("[Error] No *_frames datasets found in ./data/")
    else:
        for ds in datasets:
            process_dataset(ds)

    print("\n All frame datasets processed successfully!")
