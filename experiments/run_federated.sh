#!/bin/bash
set -e

echo "[INFO] Running federated learning simulation"

python src/run_federated.py --config configs/federated.yaml

echo "[INFO] Federated simulation finished"
