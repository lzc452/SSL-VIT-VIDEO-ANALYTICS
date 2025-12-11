import os
import cv2
from tqdm import tqdm
import multiprocessing as mp

# -------------------------------------------
# Configuration
# -------------------------------------------
DATA_ROOT = "data"
FRAME_INTERVAL = 2          # 每隔几帧提取一帧（2 表示隔帧采样）
RESIZE_SHAPE = (112, 112)   # 输出帧分辨率
NUM_WORKERS = max(1, mp.cpu_count() // 2)

# -------------------------------------------
# Extract frames from one video
# -------------------------------------------
def extract_frames_from_video(args):
    video_path, output_dir = args
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"[Error] Cannot open {video_path}"
        
        os.makedirs(output_dir, exist_ok=True)
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_INTERVAL == 0:
                frame = cv2.resize(frame, RESIZE_SHAPE)
                frame_name = f"frame_{saved_idx:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        return f"[OK] {os.path.basename(video_path)} ({saved_idx} frames)"
    except Exception as e:
        return f"[Error] {video_path}: {e}"

# -------------------------------------------
# Process one dataset
# -------------------------------------------
def process_dataset(dataset_name, is_test=False):
    input_root = os.path.join(DATA_ROOT, dataset_name)
    output_root = os.path.join(DATA_ROOT, f"{dataset_name}_frames")

    if not os.path.exists(input_root):
        print(f"[Skip] {dataset_name} not found.")
        return

    print(f"\n[Dataset] {dataset_name}")
    print(f" Input: {input_root}")
    print(f" Output: {output_root}\n")

    video_tasks = []
    def process_dir(input_dir_list): 
        # 遍历每个类别
        for cls in input_dir_list:
            cls_path = os.path.join(input_root, cls)
            if not os.path.isdir(cls_path):
                continue

            # 遍历该类下的视频
            for vid in sorted(os.listdir(cls_path)):
                if not vid.lower().endswith((".mp4", ".avi")):
                    continue
                video_path = os.path.join(cls_path, vid)
                vid_name = os.path.splitext(vid)[0]
                save_dir = os.path.join(output_root, cls, vid_name)
                if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
                    continue  # 已存在跳过
                video_tasks.append((video_path, save_dir))
    # 如果is_test 为True，只处理测试集的前3个类别
    if is_test:
        process_dir(sorted(os.listdir(input_root))[:3])
    else:
        process_dir(sorted(os.listdir(input_root)))
        
    if not video_tasks:
        print(f"[Info] No new videos to process for {dataset_name}.")
        return

    print(f"[Info] Found {len(video_tasks)} videos in {dataset_name}. Start extraction...")

    # 多进程处理
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(extract_frames_from_video, video_tasks),
                            total=len(video_tasks), ncols=80))

    # 输出统计结果
    ok = sum(1 for r in results if r.startswith("[OK]"))
    err = sum(1 for r in results if r.startswith("[Error]"))
    print(f"\n[Success] Done {dataset_name}: {ok} succeeded, {err} failed.\n")

# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    print("==========================================")
    print(" Extracting Frames from All Datasets ")
    print("==========================================")

    # DATASETS = ["UCF101", "hmdb51", "FaceForensics", "Kinetics-400-Tiny"]
    # for ds in DATASETS:
    #     process_dataset(ds)
    # process_dataset("UCF101")
    # process_dataset("hmdb51")
    # process_dataset("FaceForensics")
    process_dataset("Kinetics-400-Tiny")
    # 先拿Kinetics-400-Tiny来试试
    # process_dataset("Kinetics-400-Tiny", is_test=True)

    print("\n[Success] All datasets processed successfully!")
