# python preprocess/verify_dataset.py --clips_root data/FaceForensics_clips --splits_root data/splits/UCF101


import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("==== VERIFY DATASET START ====")


def verify_clip_shape(clip):
    """
    Acceptable shapes:
        [T, H, W, C]
        [C, T, H, W]
    """
    if clip.ndim != 4:
        return False, "Not 4D array"

    T_1 = clip.shape[0]
    T_2 = clip.shape[1]

    # Case 1: [T, H, W, C]
    if clip.shape[-1] in (1, 3) and T_1 in (8, 16, 32):
        return True, f"OK format (THWC): T={T_1}"

    # Case 2: [C, T, H, W]
    if clip.shape[0] in (1, 3) and T_2 in (8, 16, 32):
        return True, f"OK format (CTHW): T={T_2}"

    return False, f"Unexpected shape {clip.shape}"


def verify_splits(split_files, root_dir):
    errors = []
    total_refs = 0

    for split_file in split_files:
        logging.info(f"Checking split file: {split_file}")
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                total_refs += 1
                try:
                    path, label = line.split()
                except:
                    errors.append(f"BAD LINE FORMAT: {line}")
                    continue

                full_path = Path(path)
                if not full_path.exists():
                    errors.append(f"Missing clip: {path}")

    return errors, total_refs


def verify_class_distribution(clips_root):
    class_counts = {}
    for class_dir in sorted(Path(clips_root).iterdir()):
        if not class_dir.is_dir():
            continue
        clips = list(class_dir.glob("*.npy"))
        class_counts[class_dir.name] = len(clips)
    return class_counts


def verify_clips(clips_root):
    errors = []
    ok = 0
    bad = 0

    logging.info(f"Scanning clips under: {clips_root}")

    clip_paths = sorted(list(Path(clips_root).rglob("*.npy")))

    for clip_path in tqdm(clip_paths, desc="Verifying clips"):
        try:
            clip = np.load(clip_path)
        except Exception as e:
            errors.append(f"Corrupted clip: {clip_path} | error: {e}")
            bad += 1
            continue

        valid, msg = verify_clip_shape(clip)
        if not valid:
            errors.append(f"Bad shape: {clip_path} | {msg}")
            bad += 1
        else:
            ok += 1

    return ok, bad, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_root", required=True, help="Path to *_clips/ directory")
    parser.add_argument("--splits_root", required=False, help="Path to split files directory")
    parser.add_argument("--log", default="logs/verify_dataset.log")

    args = parser.parse_args()
    clips_root = Path(args.clips_root)
    splits_root = Path(args.splits_root) if args.splits_root else None

    setup_logger(args.log)

    logging.info(f"Clips root: {clips_root}")
    logging.info(f"Splits root: {splits_root}")

    # Step 1: verify all clips
    ok, bad, clip_errors = verify_clips(clips_root)

    logging.info(f"Valid clips: {ok}")
    logging.info(f"Broken clips: {bad}")

    for e in clip_errors:
        logging.error(e)

    # Step 2: verify class distribution
    class_counts = verify_class_distribution(clips_root)
    logging.info("Class clip counts:")
    for cls, count in class_counts.items():
        logging.info(f"  {cls}: {count}")

    # Step 3: verify splits
    if splits_root:
        split_files = sorted(list(Path(splits_root).glob("*_list_*.txt")))
        split_errors, total_refs = verify_splits(split_files, clips_root)
        logging.info(f"Split references checked: {total_refs}")

        if len(split_errors) > 0:
            logging.error("Split file errors:")
            for e in split_errors:
                logging.error(e)
        else:
            logging.info("Split integrity OK")

    logging.info("==== VERIFY DATASET END ====")
    print(f"Verification complete. Log written to: {args.log}")


if __name__ == "__main__":
    main()
