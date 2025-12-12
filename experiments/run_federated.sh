#!/bin/bash

# bash experiments/run_federated.sh
# # 或指定其他配置
# bash experiments/run_federated.sh configs/privacy_federated.yaml


echo "[INFO] Running federated learning simulation..."
echo "[INFO] Config: ${1:-configs/privacy_federated.yaml}"

python src/run_federated.py --config "${1:-configs/privacy_federated.yaml}"

echo "[INFO] Federated learning simulation finished."
