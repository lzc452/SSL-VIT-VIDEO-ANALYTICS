#!/bin/bash

# bash experiments/run_privacy.sh configs/privacy_visual.yaml
# bash experiments/run_privacy.sh configs/privacy_feature.yaml


echo "[INFO] Running privacy evaluation..."
echo "[INFO] Config: $1"

python src/run_privacy.py --config "$1"

echo "[INFO] Privacy evaluation finished."
