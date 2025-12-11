# preprocess/convert_hmdb_frames.py
import os
import cv2
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_video_folder(args):
    video_dir, out_root, size, overwrite = args
    rel_path = video_dir.relative_to(args_root)
    out_dir = out_root / rel_path
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted([f for f in video_dir.glob("*.jpg")], key=lambda x: int(x.stem))
    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (size, size))
        out_name = f"frame_{idx:05d}.jpg"
        out_path = out_dir / out_name
        if overwrite:
            cv2.imwrite(str(frame_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(out_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

def main():
    global args_root
    parser = argparse.ArgumentParser(description="Convert HMDB51 frames to standard format.")
    parser.add_argument("--root", required=True, help="Root path of original HMDB51 frames")
    parser.add_argument("--out", default="", help="Output root; if empty, overwrite in place")
    parser.add_argument("--size", type=int, default=112, help="Output size (default 112)")
    parser.add_argument("--workers", type=int, default=cpu_count()//2)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original frames")
    args = parser.parse_args()

    args_root = Path(args.root)
    out_root = Path(args.out) if args.out else args_root

    class_dirs = [d for d in args_root.iterdir() if d.is_dir()]
    video_dirs = []
    for c in class_dirs:
        for v in c.iterdir():
            if v.is_dir():
                video_dirs.append(v)

    tasks = [(v, out_root, args.size, args.overwrite) for v in video_dirs]
    print(f"Found {len(tasks)} videos. Starting conversion...")
    with Pool(args.workers) as p:
        list(tqdm(p.imap(process_video_folder, tasks), total=len(tasks)))

    print(f"[convert_hmdb_frames] Done. Output saved to {out_root}")

if __name__ == "__main__":
    main()
