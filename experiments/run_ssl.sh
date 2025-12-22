#!/bin/bash
set -e

echo "[INFO] Starting SSL pretraining"

python src/train_ssl.py --config configs/ssl_train.yaml

echo "[INFO] SSL pretraining finished"
