import argparse
import cv2
from pathlib import Path


def verify_frames(video_dir, min_frames, check_n=3):
    frames = sorted(video_dir.glob("*.jpg"))
    if len(frames) < min_frames:
        return False, f"too few frames ({len(frames)})"

    # sample check
    step = max(1, len(frames) // check_n)
    for f in frames[::step][:check_n]:
        img = cv2.imread(str(f))
        if img is None:
            return False, f"broken frame {f.name}"

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", type=str, required=True,
                        help="e.g. data/UCF101_frames")
    parser.add_argument("--split_file", type=str, required=True,
                        help="e.g. data/splits/UCF101_train.txt")
    parser.add_argument("--clip_len", type=int, default=16)
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    split_file = Path(args.split_file)

    bad = 0
    total = 0

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split()
            video_dir = frames_root / rel_path

            total += 1

            if not video_dir.exists():
                print(f"[ERROR] missing dir: {video_dir}")
                bad += 1
                continue

            ok, reason = verify_frames(video_dir, args.clip_len)
            if not ok:
                print(f"[ERROR] invalid video: {video_dir} ({reason})")
                bad += 1

    print(f"[INFO] verify finished: total={total}, bad={bad}")
    if bad > 0:
        print("[WARN] dataset has invalid samples")
    else:
        print("[INFO] dataset verified successfully")


if __name__ == "__main__":
    main()
