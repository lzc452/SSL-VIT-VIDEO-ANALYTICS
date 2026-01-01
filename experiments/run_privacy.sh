#!/bin/bash
set -e

echo "[INFO] Running privacy evaluation (visual + feature)"
python src/run_privacy.py --config configs/privacy.yaml
echo "[INFO] Privacy evaluation finished"
