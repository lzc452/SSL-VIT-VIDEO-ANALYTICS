#!/bin/bash
set -e

echo "[INFO] Running ablation and sensitivity studies"

python src/ablations.py

echo "[INFO] Ablation experiments finished"
