#!/bin/bash

echo "[INFO] Running dynamic inference evaluation..."
echo "[INFO] Config: configs/dynamic_infer.yaml"

python src/run_dynamic.py --config configs/dynamic_infer.yaml

echo "[INFO] Dynamic inference evaluation finished."
