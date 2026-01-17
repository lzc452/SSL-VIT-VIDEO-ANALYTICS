#!/usr/bin/env bash
set -e

CFG="configs/mae_train.yaml"

python -m src.mae.train_mae --config "${CFG}"
