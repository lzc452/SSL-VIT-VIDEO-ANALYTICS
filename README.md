## Overview
This repository contains the full experimental framework developed for the master’s research project:
“Self-Supervised Lightweight Transformers for Privacy-Aware Video Analytics at the Edge.”

The project investigates lightweight, privacy-preserving Transformer architectures for video analytics under constrained edge conditions.
It integrates three hierarchical privacy layers — visual, feature-level, and system-level (federated) — within a self-supervised learning pipeline.

## Research Objectives
1. Develop a self-supervised lightweight Transformer architecture capable of efficient representation learning on edge-scale video data.

2. Integrate privacy-preserving mechanisms (visual obfuscation, feature perturbation, and federated learning) into the inference pipeline for secure video analytics.

3. Evaluate performance trade-offs among accuracy, latency, and privacy under constrained GPU resources.

## System Architecture
```bash
Input Video
   │
   ├──► Visual Privacy (face masking)
   │
   ├──► Self-Supervised Lightweight Transformer
   │        └─ Feature-Level Perturbation (z' = z + N(0,σ²I))
   │
   ├──► Federated Learning (system-level decentralisation)
   │
   └──► Output: Privacy-Preserved Action Recognition Results
```

Each module corresponds to one level of the privacy-aware hierarchy:
| Level               | Mechanism                                | File / Script                     | Purpose                          |
| ------------------- | ---------------------------------------- | --------------------------------- | -------------------------------- |
| **Visual Privacy**  | Gaussian face masking (YuNet)            | `src/privacy/visual_mask.py`      | Removes visual identifiers       |
| **Feature Privacy** | Feature perturbation (z' = z + N(0,σ²I)) | `src/privacy/feature_noise.py`    | Prevents feature leakage         |
| **System Privacy**  | Federated learning decentralisation      | `src/privacy/system_federated.py` | Prevents raw data centralisation |

## Project Structure
```bash
SSL-VIT-VIDEO-ANALYTICS/
├─ configs/               # YAML configuration files
├─ data/                  # Datasets and train/val/test splits
├─ preprocess/            # Frame extraction & sampling scripts
├─ src/                   # Core modules (model, training, privacy, federated)
├─ experiments/           # Shell scripts for batch execution
├─ results/               # Experiment outputs (csv, figures, models)
├─ logs/                  # Training and evaluation logs
├─ notebooks/             # Jupyter/Colab notebooks for analysis
├─ requirements.txt       # Dependency list
├─ README.md              # This file
└─ .gitignore             # Ignore large or temporary files
```

## Installation
### Clone Repository
```bash
git clone https://github.com/lzc452/SSL-VIT-VIDEO-ANALYTICS.git
cd SSL-VIT-VIDEO-ANALYTICS
```
### Start Training
```bash
cd ThesisExp
python src/train_ssl.py --config configs/ssl_train.yaml
python src/train_finetune.py --config configs/finetune.yaml
```
### Run Privacy Experiments
Unified execution:
```bash
bash experiments/run_privacy.sh
```
Or separately:
```bash
python src/privacy/visual_mask.py
python src/privacy/feature_noise.py
python src/privacy/system_federated.py
```
## Federated Learning Integration
Federated simulation runs multiple local clients (3–5) with FedAvg aggregation.
```bash
python src/federated/fed_loop.py --clients 3 --rounds 5
```
Metrics generated:
1. Accuracy vs Communication Rounds
2. Communication Volume (MB/round)
3. R* (rounds to reach target accuracy)

Output path:
```bash
results/privacy/federated/
 ├─ accuracy_vs_round.csv
 ├─ comm_volume.csv
 └─ fed_curve.png
```
## Visualization and Result Export
All experiments output .csv logs and .png figures under /results/.
To generate paper-quality figures:
```bash
python src/plotting.py
```
Outputs:
```bash
results/figures/
 ├─ latency_accuracy_frontier.png
 ├─ privacy_tradeoff_heatmap.png
 ├─ federated_curve.png
 └─ sampling_ablation.png
```
## Estimated Resource Plan
| Task                     | GPU             | Runtime | Cost (Vast.ai / RunPod) |
| ------------------------ | --------------- | ------- | ----------------------- |
| SSL Pretraining          | RTX 4090        | 3–5 hrs | $2–3                    |
| Fine-tuning              | RTX 4090        | 2 hrs   | $1.5                    |
| Dynamic Inference        | L4 / T4         | 1 hr    | $0.7                    |
| Privacy (Visual/Feature) | T4 / 3090       | 1–2 hrs | $0.8                    |
| Federated Simulation     | A4000           | 2 hrs   | $1.5                    |
| **Total**                | —               | ≈12 hrs | **$7–10 USD**           |

## Log & Result Management
All experiment runs automatically record:
```bash
logs/
├─ train_ssl.log
├─ finetune.log
├─ privacy.log
├─ federated.log
```
Aggregated results saved to:
```bash
results/summary.csv
```
## Ethical & Privacy Notes
1. Only publicly available datasets are used.
2. All privacy experiments (masking, feature noise, federated) operate on anonymised or synthetic data.
3. No identifiable human data is collected or redistributed.
4. The system design adheres to GDPR-like privacy-by-design principles.
## Scripts and Automation
| Purpose            | Script                        | Description                             |
| ------------------ | ----------------------------- | --------------------------------------- |
| Pretraining        | `experiments/run_ssl.sh`      | Run self-supervised learning            |
| Finetuning         | `experiments/run_finetune.sh` | Run supervised fine-tuning              |
| Dynamic Inference  | `experiments/run_dynamic.sh`  | Latency–accuracy trade-off              |
| Privacy Evaluation | `experiments/run_privacy.sh`  | Run visual, feature & federated privacy |
| Ablation Study     | `experiments/run_ablation.sh` | Sampling & parameter sensitivity        |

## Citation
If you use this repository or its methodology, please cite:
```rust
@misc{zhicai2025privacyaware,
  title={Self-Supervised Lightweight Transformers for Privacy-Aware Video Analytics at the Edge},
  author={Li, Zhicai},
  year={2025},
  institution={Universiti Kebangsaan Malaysia},
  note={Master’s Thesis Project Repository}
}
```

## Author & Supervision
Author: Li Zhicai (UKM, Master of Computer Science)
Supervisor: [Prof. Dr. SITI NORUL HUDA], Cyber Analytics Lab, Universiti Kebangsaan Malaysia
Contact: [p146924@siswa.ukm.edu.my]

## License
This repository is for academic research and educational purposes.
Commercial use or redistribution of modified datasets is prohibited.