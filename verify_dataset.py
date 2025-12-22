#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
from tqdm import tqdm


def load_split_file(split_path):
    samples = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.split()
            samples.append((path, label))
    return samples


def save_split_file(split_path, samples):
    with open(split_path, "w") as f:
        for path, label in samples:
            f.write(f"{path} {label}\n")


def has_valid_frames(frame_dir, min_frames):
    if not os.path.isdir(frame_dir):
        return False
    frames = [
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]
    return len(frames) >= min_frames


def clean_splits(split_files, min_frames, dry_run=False):
    """
    Remove samples whose frame directory is empty or insufficient.
    """
    removed = defaultdict(list)
    kept = {}

    for split_path in split_files:
        print(f"[INFO] Cleaning split: {split_path}")
        samples = load_split_file(split_path)

        new_samples = []
        for frame_dir, label in tqdm(samples, desc=f"Checking {os.path.basename(split_path)}"):
            if has_valid_frames(frame_dir, min_frames):
                new_samples.append((frame_dir, label))
            else:
                removed[split_path].append(frame_dir)

        kept[split_path] = new_samples

        print(
            f"[INFO] {os.path.basename(split_path)}: "
            f"kept={len(new_samples)}, removed={len(removed[split_path])}"
        )

        if not dry_run:
            save_split_file(split_path, new_samples)

    return kept, removed


def check_length(before, after):
    """
    Check whether sample counts are consistent after cleaning.
    """
    print("[INFO] Checking split length consistency")
    for split_path in before:
        n_before = len(before[split_path])
        n_after = len(after.get(split_path, []))

        if n_after > n_before:
            print(
                f"[ERROR] {os.path.basename(split_path)}: "
                f"after-clean ({n_after}) > before-clean ({n_before})"
            )
        else:
            print(
                f"[INFO] {os.path.basename(split_path)}: "
                f"{n_before} -> {n_after} samples"
            )


def verify_only(split_files, min_frames):
    """
    Only report invalid samples without modifying splits.
    """
    total_invalid = 0

    for split_path in split_files:
        print(f"[INFO] Verifying split: {split_path}")
        samples = load_split_file(split_path)

        invalid = []
        for frame_dir, _ in tqdm(samples, desc=f"Verifying {os.path.basename(split_path)}"):
            if not has_valid_frames(frame_dir, min_frames):
                invalid.append(frame_dir)

        print(
            f"[INFO] {os.path.basename(split_path)}: "
            f"invalid={len(invalid)}, total={len(samples)}"
        )

        total_invalid += len(invalid)

    print(f"[INFO] Total invalid samples across splits: {total_invalid}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and optionally clean dataset splits (Lazy Frame Mode)"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing split txt files"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=16,
        help="Minimum number of frames required per video"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove invalid samples from split files"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report cleaning results without modifying files"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    split_files = [
        os.path.join(args.splits_dir, f)
        for f in os.listdir(args.splits_dir)
        if f.endswith(".txt")
    ]
    split_files.sort()

    if not split_files:
        print("[ERROR] No split files found")
        return

    print(f"[INFO] Found {len(split_files)} split files")

    if not args.clean:
        verify_only(split_files, args.min_frames)
        return

    # Load before-clean statistics
    before = {
        path: load_split_file(path)
        for path in split_files
    }

    after, removed = clean_splits(
        split_files,
        min_frames=args.min_frames,
        dry_run=args.dry_run
    )

    check_length(before, after)

    # Summary
    print("[INFO] Cleaning summary")
    for split_path, removed_list in removed.items():
        print(
            f"[INFO] {os.path.basename(split_path)}: "
            f"removed {len(removed_list)} invalid samples"
        )


if __name__ == "__main__":
    main()
