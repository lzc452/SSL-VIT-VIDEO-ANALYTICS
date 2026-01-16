import os
from pathlib import Path
import random

def generate_mixed_list():
    # 定义三个数据集提取帧后的根目录
    # 请根据你的实际存储路径修改
    dataset_configs = [
        {"root": "data/UCF101_frames", "name": "UCF101"},
        {"root": "data/hmdb51_frames", "name": "hmdb51"},
        {"root": "data/Kinetics-400-Tiny_frames", "name": "Kinetics-400-Tiny"}
    ]
    
    output_file = Path("data/mixed_train_list.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    mixed_samples = []

    for config in dataset_configs:
        root_path = Path(config["root"])
        if not root_path.exists():
            print(f"[WARN] {config['name']} path not found: {root_path}")
            continue
            
        print(f"[INFO] Processing {config['name']}...")
        
        # 遍历类别目录 (e.g., data/UCF101_frames/ApplyEyeMakeup)
        for class_dir in root_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            # 遍历视频文件夹 (e.g., data/UCF101_frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01)
            for video_dir in class_dir.iterdir():
                if video_dir.is_dir():
                    # 检查文件夹内是否有 jpg 文件
                    if any(video_dir.glob("*.jpg")):
                        # 记录相对路径和占位标签
                        mixed_samples.append(f"{video_dir} 0")

    # 随机打乱列表，有助于 SSL 训练的收敛
    random.shuffle(mixed_samples)
    
    with open(output_file, "w") as f:
        for sample in mixed_samples:
            f.write(f"{sample}\n")
            
    print(f"[SUCCESS] Total samples found: {len(mixed_samples)}")
    print(f"[SUCCESS] Saved to: {output_file}")

if __name__ == "__main__":
    generate_mixed_list()