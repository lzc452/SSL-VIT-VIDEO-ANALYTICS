#!/bin/bash
set -e

# Journal-grade dynamic inference experiment runner.
# It runs three modes sequentially and writes CSVs into separate folders.

BASE_CFG="configs/base.yaml"
DYN_CFG="configs/dynamic.yaml"

echo "[INFO] Running dynamic inference experiments"
echo "[INFO] Base cfg: ${BASE_CFG}"
echo "[INFO] Dynamic cfg: ${DYN_CFG}"

python src/run_dynamic.py --base "${BASE_CFG}" --cfg "${DYN_CFG}" --mode early_exit   --save_dir results/dynamic/early_exit
python src/run_dynamic.py --base "${BASE_CFG}" --cfg "${DYN_CFG}" --mode frame_gating --save_dir results/dynamic/frame_gating
python src/run_dynamic.py --base "${BASE_CFG}" --cfg "${DYN_CFG}" --mode hybrid      --save_dir results/dynamic/hybrid

echo "[INFO] Done. CSVs saved under results/dynamic/{early_exit,frame_gating,hybrid}/"
