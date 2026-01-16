#!/bin/bash
set -e

echo "[INFO] Starting MAE SSL pretraining"

python src/train_ssl_mae.py --config configs/ssl_mae.yaml

echo "[INFO] MAE SSL pretraining finished"
