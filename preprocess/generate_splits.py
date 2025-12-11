import os
import random
from tqdm import tqdm

# -------------------------------------------
# Configuration
# -------------------------------------------
DATA_ROOT = "data"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train / validation / test
SEED = 42
random.seed(SEED)

# -------------------------------------------
# Helper: load class index mapping
# -------------------------------------------
def load_class_index(dataset_name):
    """
    Read class mapping from e.g., data/UCF101/ucf101_class_index.txt
    """
    index_file = os.path.join(DATA_ROOT, dataset_name.replace("_clips", ""), f"{dataset_name.replace('_clips', '').lower()}_class_index.txt")
    mapping = {}
    if not os.path.exists(index_file):
        print(f"[Warn] No class index file for {dataset_name}. Using alphabetical order.")
        return None
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                cls, idx = parts
                mapping[cls] = int(idx)
    return mapping

# -------------------------------------------
# Process one dataset
# -------------------------------------------
def process_dataset(clips_dataset):
    input_root = os.path.join(DATA_ROOT, clips_dataset)
    split_dir = os.path.join(DATA_ROOT, "splits")
    os.makedirs(split_dir, exist_ok=True)

    dataset_name = clips_dataset.replace("_clips", "").lower()
    class_map = load_class_index(clips_dataset)
    if class_map is None:
        # 如果没有映射文件，自动按字母顺序分配标签
        classes = sorted([c for c in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, c))])
        class_map = {cls: idx for idx, cls in enumerate(classes)}

    # 收集所有 clip 路径
    all_clips = []
    for cls in tqdm(sorted(class_map.keys()), desc=f"[Scanning {clips_dataset}]"):
        cls_path = os.path.join(input_root, cls)
        if not os.path.isdir(cls_path): continue

        for vid in sorted(os.listdir(cls_path)):
            vid_path = os.path.join(cls_path, vid)
            if not os.path.isdir(vid_path): continue
            for clip_file in os.listdir(vid_path):
                if clip_file.endswith(".npy"):
                    clip_path = os.path.join(clips_dataset, cls, vid, clip_file)
                    all_clips.append((clip_path, class_map[cls]))

    if not all_clips:
        print(f"[Info] No clips found in {clips_dataset}, skipping.")
        return

    # 随机打乱
    random.shuffle(all_clips)
    n_total = len(all_clips)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])
    n_test = n_total - n_train - n_val

    splits = {
        "train": all_clips[:n_train],
        "val": all_clips[n_train:n_train+n_val],
        "test": all_clips[n_train+n_val:]
    }

    # 保存到 ./data/splits/
    for split_name, items in splits.items():
        out_file = os.path.join(split_dir, f"{split_name}_list_{dataset_name}.txt")
        if os.path.exists(out_file):
            print(f"[Skip] {out_file} already exists.")
            continue
        with open(out_file, "w", encoding="utf-8") as f:
            for path, label in items:
                f.write(f"{path} {label}\n")
        print(f"[Saved] {out_file} ({len(items)} clips)")

    print(f"[Success] Done {clips_dataset}: {n_train} train / {n_val} val / {n_test} test\n")

# -------------------------------------------
# Main: Auto-detect all *_clips datasets
# -------------------------------------------
if __name__ == "__main__":
    print("==========================================")
    print(" Generating Train/Val/Test Splits ")
    print("==========================================")

    datasets = [d for d in os.listdir(DATA_ROOT)
                if d.endswith("_clips") and os.path.isdir(os.path.join(DATA_ROOT, d))]

    if not datasets:
        print("[Error] No *_clips datasets found in ./data/")
    else:
        for ds in datasets:
            process_dataset(ds)

    print("\n All dataset splits generated successfully!")
