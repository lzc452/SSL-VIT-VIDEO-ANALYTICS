#!/bin/bash
set -e

echo "[INFO] Running full dataset pipeline (Frame-Lazy)"

DATASET=UCF101
FRAMES_ROOT=data/${DATASET}_frames
SPLIT_ROOT=data/splits

echo "[INFO] Generating lazy splits"
python preprocess/generate_splits_lazy.py --frames_root ${FRAMES_ROOT} --output_dir ${SPLIT_ROOT}

echo "[INFO] Verifying dataset"
python preprocess/verify_dataset.py --frames_root ${FRAMES_ROOT} --split_file ${SPLIT_ROOT}/${DATASET}_train.txt --clip_len 16

echo "[INFO] Dataset pipeline finished successfully"
