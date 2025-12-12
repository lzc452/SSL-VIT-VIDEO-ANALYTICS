#!/bin/bash

echo "[INFO] Running SSL pretraining..."
echo "[INFO] Config: configs/ssl_train.yaml"

python src/train_ssl.py --config configs/ssl_train.yaml

echo "[INFO] SSL pretraining finished."
