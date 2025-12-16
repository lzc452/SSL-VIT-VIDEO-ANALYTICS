import argparse
import cv2
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


def extract_one_video(args):
    video_path, out_dir, size = args
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        return "[INFO] skip existing", str(video_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "[ERROR] cannot open", str(video_path)

    idx = 0
    success = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
        except Exception:
            success = False
            break

        fname = out_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(fname), frame)
        idx += 1

    cap.release()

    if idx == 0 or not success:
        # cleanup
        for f in out_dir.glob("*.jpg"):
            f.unlink()
        out_dir.rmdir()
        return "[ERROR] failed extract", str(video_path)

    return "[INFO] done", f"{video_path} ({idx} frames)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True,
                        help="e.g. data/UCF101")
    parser.add_argument("--output_root", type=str, required=True,
                        help="e.g. data/UCF101_frames")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)

    jobs = []

    for cls_dir in sorted(video_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for vid in cls_dir.glob("*.mp4"):
            out_dir = output_root / cls_dir.name / vid.stem
            jobs.append((vid, out_dir, args.image_size))

    print(f"[INFO] Total videos: {len(jobs)}")

    with mp.Pool(args.workers) as pool:
        for msg, info in tqdm(pool.imap_unordered(extract_one_video, jobs), total=len(jobs)):
            if msg.startswith("[ERROR]"):
                print(msg, info)

    print("[INFO] extract_frames finished")


if __name__ == "__main__":
    main()
