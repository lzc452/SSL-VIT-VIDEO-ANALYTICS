#!/bin/bash

echo "[INFO] Running supervised fine-tuning..."
echo "[INFO] Config: configs/finetune.yaml"

python src/train_finetune.py --config configs/finetune.yaml

echo "[INFO] Fine-tuning finished."
