#!/bin/bash
set -e

echo "[INFO] Running supervised fine-tuning (4-mode comparison)..."
echo "[INFO] Config: configs/finetune.yaml"

MODES=("ft_random" "linear_probe" "ft_ssl" "two_stage")

mkdir -p logs

for MODE in "${MODES[@]}"; do
  echo "----------------------------------------"
  echo "[INFO] Mode = ${MODE}"
  echo "----------------------------------------"

  python src/train_finetune.py \
    --config configs/finetune.yaml \
    --mode ${MODE} \
    2>&1 | tee logs/finetune_${MODE}.log
done

echo "[INFO] All fine-tuning modes finished."
