#!/bin/bash
set -e  # stop on error

# ================================
#     Full Pipeline From Frames
# ================================
# Usage:
#   bash preprocess/full_pipeline_dataset.sh UCF101
#
# Steps:
#   1. frames  → make_clips
#   2. clips   → generate_splits
#   3. verify  → verify_dataset
#
# All logs saved under logs/
# ================================

DATASET_NAME=$1

if [ -z "$DATASET_NAME" ]; then
    echo "ERROR: Dataset name not provided."
    echo "Usage: bash preprocess/full_pipeline_dataset.sh <DATASET_NAME>"
    exit 1
fi

FRAMES_ROOT="data/${DATASET_NAME}_frames"
CLIPS_ROOT="data/${DATASET_NAME}_clips"
SPLITS_ROOT="data/splits/${DATASET_NAME}"

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "=============================================="
echo "FULL PIPELINE START for dataset: ${DATASET_NAME}"
echo "=============================================="
echo ""

echo "Frames directory      : ${FRAMES_ROOT}"
echo "Clips output directory: ${CLIPS_ROOT}"
echo "Splits directory      : ${SPLITS_ROOT}"
echo ""

# ------------------------------
# Step 1: Make Clips
# ------------------------------
echo "=============================================="
echo "Step 1/3: Generating clips from frames..."
echo "=============================================="

python preprocess/make_clips.py \
    --root ${FRAMES_ROOT} \
    --out ${CLIPS_ROOT} \
    --size 112 \
    --clip_len 16 \
    --stride 8 \
    | tee ${LOG_DIR}/make_clips_${DATASET_NAME}.log

echo "Clips generated at: ${CLIPS_ROOT}"
echo ""

# ------------------------------
# Step 2: Generate Splits
# ------------------------------
echo "=============================================="
echo "Step 2/3: Generating train/val/test splits..."
echo "=============================================="

mkdir -p ${SPLITS_ROOT}

python preprocess/generate_splits.py \
    --clips_root ${CLIPS_ROOT} \
    --out ${SPLITS_ROOT} \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    | tee ${LOG_DIR}/splits_${DATASET_NAME}.log

echo "Splits generated at: ${SPLITS_ROOT}"
echo ""

# ------------------------------
# Step 3: Verify Dataset
# ------------------------------
echo "=============================================="
echo "Step 3/3: Verifying dataset integrity..."
echo "=============================================="

python preprocess/verify_dataset.py \
    --clips_root ${CLIPS_ROOT} \
    --splits_root ${SPLITS_ROOT} \
    --log ${LOG_DIR}/verify_${DATASET_NAME}.log

echo ""
echo "Dataset verification complete."
echo "Log file: ${LOG_DIR}/verify_${DATASET_NAME}.log"

echo ""
echo "=============================================="
echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="
