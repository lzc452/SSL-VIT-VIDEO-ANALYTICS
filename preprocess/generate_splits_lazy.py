# preprocess/generate_splits_lazy.py

import argparse
import random
from pathlib import Path

# 输入：
# data/UCF101_frames/
#    ApplyEyeMakeup/
#       v_xxx/
#          frame_00001.jpg
#          frame_00002.jpg
#          ...
# 输出
# data/splits/
#    UCF101_train.txt
#    UCF101_val.txt
#    UCF101_test.txt
# 格式
# <path_to_video_frame_folder> <label>
# e.g.
# data/UCF101_frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", type=str, required=True,
                        help="Root directory of extracted frames, e.g. data/UCF101_frames")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    class_to_idx = {cls.name: i for i, cls in enumerate(classes)}

    train_list, val_list, test_list = [], [], []

    for cls in classes:
        videos = sorted([v for v in cls.iterdir() if v.is_dir()])
        random.shuffle(videos)

        n = len(videos)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]

        label = class_to_idx[cls.name]

        for v in train_videos:
            train_list.append(f"{v.as_posix()} {label}\n")
        for v in val_videos:
            val_list.append(f"{v.as_posix()} {label}\n")
        for v in test_videos:
            test_list.append(f"{v.as_posix()} {label}\n")

    dataset_name = frames_root.name.replace("_frames", "")
    (output_dir / f"{dataset_name}_train.txt").write_text("".join(train_list))
    (output_dir / f"{dataset_name}_val.txt").write_text("".join(val_list))
    (output_dir / f"{dataset_name}_test.txt").write_text("".join(test_list))

    print("[INFO] Split generation completed")
    print(f"[INFO] Train: {len(train_list)} samples")
    print(f"[INFO] Val:   {len(val_list)} samples")
    print(f"[INFO] Test:  {len(test_list)} samples")


if __name__ == "__main__":
    main()
