import os
import subprocess
import re
import shutil
import concurrent.futures
from tqdm import tqdm

# -------------------------------------------
# Basic tools
# -------------------------------------------
def normalize_name(name: str):
    name = name.lower()
    name = re.sub(r"[()!.,'\"\[\]{}]", "", name)
    name = re.sub(r"[\s\-]+", "_", name)
    return name.strip("_")

def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    candidates = [
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"/usr/bin/ffmpeg",
        r"/usr/local/bin/ffmpeg"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("FFmpeg not found. Please install or update PATH.")

# -------------------------------------------
# Convert one video
# -------------------------------------------
def convert_video(args):
    ffmpeg_bin, src, dst = args
    if os.path.exists(dst):
        return f"Skip: {os.path.basename(dst)}"
    try:
        subprocess.run(
            [ffmpeg_bin, "-loglevel", "error", "-y", "-i", src,
             "-c:v", "libx264", "-crf", "23", dst],
            check=True
        )
        os.remove(src)
        return f"OK: {os.path.basename(dst)}"
    except Exception as e:
        return f"Error: {os.path.basename(src)} -> {e}"

# -------------------------------------------
# Rename all folders to normalized names
# -------------------------------------------
def normalize_dirnames(root):
    for dataset in os.listdir(root):
        ds_path = os.path.join(root, dataset)
        if not os.path.isdir(ds_path): continue
        print(f"\n[Dataset] {dataset}")
        for cls in os.listdir(ds_path):
            class_path = os.path.join(ds_path, cls)
            if not os.path.isdir(class_path): continue
            new_name = normalize_name(cls)
            new_path = os.path.join(ds_path, new_name)
            if class_path != new_path:
                os.rename(class_path, new_path)
                print(f"  Renamed: {cls} -> {new_name}")

# -------------------------------------------
# Parallel conversion
# -------------------------------------------
def convert_to_mp4_parallel(root, workers=None):
    ffmpeg_bin = find_ffmpeg()
    print(f"\n[FFmpeg] Using binary: {ffmpeg_bin}")

    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    print(f"[Parallel] Using {workers} workers\n")

    tasks = []
    for dataset in os.listdir(root):
        ds_path = os.path.join(root, dataset)
        if not os.path.isdir(ds_path): continue
        for cls in os.listdir(ds_path):
            class_path = os.path.join(ds_path, cls)
            if not os.path.isdir(class_path): continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(".avi"):
                    src = os.path.join(class_path, fname)
                    dst = src.rsplit(".", 1)[0] + ".mp4"
                    tasks.append((ffmpeg_bin, src, dst))

    # No tasks
    if not tasks:
        print("No .avi files found. Skipping conversion.")
        return

    print(f"[Info] Found {len(tasks)} videos to convert.\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(convert_video, tasks), total=len(tasks)))

    # Report summary
    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("Skip"))
    err = sum(1 for r in results if r.startswith("Error"))
    print(f"\n Conversion complete: {ok} OK, {skip} skipped, {err} errors.")

# -------------------------------------------
# Generate class index
# -------------------------------------------
def generate_class_index(root):
    for dataset in os.listdir(root):
        ds_path = os.path.join(root, dataset)
        if not os.path.isdir(ds_path): continue
        class_list = sorted([d for d in os.listdir(ds_path)
                             if os.path.isdir(os.path.join(ds_path, d))])
        if not class_list: continue
        out_path = os.path.join(ds_path, f"{dataset.lower()}_class_index.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, cls in enumerate(class_list):
                f.write(f"{cls} {idx}\n")
        print(f"[Saved] Class index → {out_path} ({len(class_list)} classes)")

# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    ROOT = "dataset"

    print("==========================================")
    print(" Standardizing Dataset Structures (Parallel)")
    print("==========================================")

    normalize_dirnames(ROOT)
    convert_to_mp4_parallel(ROOT)
    generate_class_index(ROOT)

    print("\n All datasets standardized successfully!")
