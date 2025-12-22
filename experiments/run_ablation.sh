#!/bin/bash
set -e

echo "[INFO] Running ablation and sensitivity studies"

python src/train_finetune.py \
  --config configs/ablation.yaml

echo "[INFO] Ablation experiments finished"
