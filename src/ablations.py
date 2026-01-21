import os
import copy
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "configs"

BASE_YAML = CFG_DIR / "base.yaml"
SSL_YAML  = CFG_DIR / "ssl_train.yaml"
FT_YAML   = CFG_DIR / "finetune.yaml"
PRIV_YAML = CFG_DIR / "privacy.yaml"
DYN_YAML  = CFG_DIR / "dynamic.yaml"

OUT_ROOT = REPO_ROOT / "results" / "ablation_runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def read_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data

def write_yaml(p: Path, d: dict):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)

def deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def run(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
        return_code = p.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(cmd)}")

def main():
    # ---------- backup original yamls ----------
    base0 = read_yaml(BASE_YAML)
    ssl0  = read_yaml(SSL_YAML)
    ft0   = read_yaml(FT_YAML)
    priv0 = read_yaml(PRIV_YAML)
    dyn0  = read_yaml(DYN_YAML)

    meta_all = []

    try:
        # ============================================================
        # ABLATION SET 1: Data sampling sensitivity (base.yaml)
        # (clip_len / stride / image_size) -> run finetune(two_stage) only
        # ============================================================
        sampling_grid = [
            {"name": "SAMP_CL16_ST4_IM112", "base": {"dataset": {"clip_len": 16, "stride": 4, "image_size": 112}}},
            {"name": "SAMP_CL32_ST4_IM112", "base": {"dataset": {"clip_len": 32, "stride": 4, "image_size": 112}}},
            {"name": "SAMP_CL32_ST2_IM112", "base": {"dataset": {"clip_len": 32, "stride": 2, "image_size": 112}}},
            {"name": "SAMP_CL32_ST8_IM112", "base": {"dataset": {"clip_len": 32, "stride": 8, "image_size": 112}}},
            {"name": "SAMP_CL32_ST4_IM96",  "base": {"dataset": {"clip_len": 32, "stride": 4, "image_size": 96}}},
            {"name": "SAMP_CL32_ST4_IM128", "base": {"dataset": {"clip_len": 32, "stride": 4, "image_size": 128}}},
        ]

        for exp in sampling_grid:
            tag = exp["name"]
            save_dir = OUT_ROOT / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            # patch base.yaml
            base_cfg = copy.deepcopy(base0)
            deep_update(base_cfg, exp["base"])
            write_yaml(BASE_YAML, base_cfg)

            # patch finetune.yaml save_dir to avoid overwrite
            ft_cfg = copy.deepcopy(ft0)
            ft_cfg["paths"]["save_dir"] = str(save_dir / "finetune")
            write_yaml(FT_YAML, ft_cfg)

            # run finetune two_stage (best reference)
            log_path = save_dir / "logs" / "finetune_two_stage.log"
            run(["python", "src/train_finetune.py", "--config", "configs/finetune.yaml", "--mode", "two_stage"], log_path)

            meta_all.append({"tag": tag, "type": "sampling", "overrides": exp})

        # ============================================================
        # ABLATION SET 2: SSL objective knobs (ssl_train.yaml)
        # Minimal but journal-meaningful:
        #   (a) MFM-only (top_weight=0)
        #   (b) MFM+TOP baseline
        #   (c) MFM+TOP with detach backbone (TOP only trains head)
        # Each: run short SSL (e.g., 20 epochs) + then finetune(two_stage)
        # ============================================================
        ssl_grid = [
            {"name": "SSL_MFM_ONLY_m075", "ssl": {"training": {"epochs": 20}, "ssl_objectives": {"mask_ratio": 0.75, "top_weight": 0.0}}},
            {"name": "SSL_MFM_TOP_m075",  "ssl": {"training": {"epochs": 20}, "ssl_objectives": {"mask_ratio": 0.75, "top_weight": 1.0, "top_detach_backbone": False}}},
            {"name": "SSL_MFM_TOP_DETACH_m075", "ssl": {"training": {"epochs": 20}, "ssl_objectives": {"mask_ratio": 0.75, "top_weight": 1.0, "top_detach_backbone": True}}},
        ]

        for exp in ssl_grid:
            tag = exp["name"]
            save_dir = OUT_ROOT / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            # restore base (keep your default sampling for SSL ablation)
            write_yaml(BASE_YAML, base0)

            # patch ssl_train.yaml save_dir per run
            ssl_cfg = copy.deepcopy(ssl0)
            deep_update(ssl_cfg, exp["ssl"])
            ssl_cfg["training"]["save_dir"] = str(save_dir / "ssl_pretrain")
            write_yaml(SSL_YAML, ssl_cfg)

            # run SSL
            log_path = save_dir / "logs" / "ssl.log"
            run(["python", "src/train_ssl.py"], log_path)

            # pick latest checkpoint as pretrained_ssl
            # (your SSL saves ssl_ep{ep}.pth; use last ep)
            ep = int(ssl_cfg["training"]["epochs"])
            ckpt = save_dir / "ssl_pretrain" / f"ssl_ep{ep}.pth"

            # patch finetune.yaml to point to this SSL ckpt & separate save dir
            ft_cfg = copy.deepcopy(ft0)
            ft_cfg["model"]["pretrained_ssl"] = str(ckpt)
            ft_cfg["paths"]["save_dir"] = str(save_dir / "finetune")
            write_yaml(FT_YAML, ft_cfg)

            # run finetune two_stage
            log_path = save_dir / "logs" / "finetune_two_stage.log"
            run(["python", "src/train_finetune.py", "--config", "configs/finetune.yaml", "--mode", "two_stage"], log_path)

            meta_all.append({"tag": tag, "type": "ssl_objective", "overrides": exp})

        # ============================================================
        # ABLATION SET 3: Privacy strength scan (privacy.yaml)
        # - visual blur_kernel levels
        # - feature noise_sigmas already supports list
        # ============================================================
        privacy_grid = [
            {"name": "PRIV_VIS_BLUR_15", "priv": {"visual_privacy": {"blur_kernel": 15}}},
            {"name": "PRIV_VIS_BLUR_31", "priv": {"visual_privacy": {"blur_kernel": 31}}},
            {"name": "PRIV_VIS_BLUR_51", "priv": {"visual_privacy": {"blur_kernel": 51}}},
        ]

        for exp in privacy_grid:
            tag = exp["name"]
            save_dir = OUT_ROOT / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            # patch privacy.yaml save dir
            priv_cfg = copy.deepcopy(priv0)
            deep_update(priv_cfg, exp["priv"])
            priv_cfg["output"]["save_dir"] = str(save_dir / "privacy")
            write_yaml(PRIV_YAML, priv_cfg)

            log_path = save_dir / "logs" / "privacy.log"
            run(["python", "src/run_privacy.py", "--config", "configs/privacy.yaml"], log_path)

            meta_all.append({"tag": tag, "type": "privacy", "overrides": exp})

        # ============================================================
        # ABLATION SET 4: Dynamic inference knob scan (dynamic.yaml)
        # Run all 3 modes (early_exit/frame_gating/hybrid) with different lists
        # ============================================================
        dyn_grid = [
            {"name": "DYN_THRESH_TIGHT", "dyn": {"dynamic": {"confidence_thresholds": [0.75, 0.80, 0.85, 0.90]}}},
            {"name": "DYN_THRESH_LOOSE", "dyn": {"dynamic": {"confidence_thresholds": [0.55, 0.60, 0.65, 0.70]}}},
            {"name": "DYN_GATING_TOPK",   "dyn": {"dynamic": {"gating_topk_list": [4, 6, 8, 10, 12, 16]}}},
        ]

        for exp in dyn_grid:
            tag = exp["name"]
            save_dir = OUT_ROOT / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            dyn_cfg = copy.deepcopy(dyn0)
            deep_update(dyn_cfg, exp["dyn"])
            write_yaml(DYN_YAML, dyn_cfg)

            # run 3 modes (same as experiments/run_dynamic.sh but redirect save_dir)
            log_path = save_dir / "logs" / "dynamic.log"
            cmd = [
                "bash", "-lc",
                f"python src/run_dynamic.py --base configs/base.yaml --cfg configs/dynamic.yaml --mode early_exit   --save_dir {save_dir/'dynamic/early_exit'} && "
                f"python src/run_dynamic.py --base configs/base.yaml --cfg configs/dynamic.yaml --mode frame_gating --save_dir {save_dir/'dynamic/frame_gating'} && "
                f"python src/run_dynamic.py --base configs/base.yaml --cfg configs/dynamic.yaml --mode hybrid      --save_dir {save_dir/'dynamic/hybrid'}"
            ]
            run(cmd, log_path)

            meta_all.append({"tag": tag, "type": "dynamic", "overrides": exp})

        # ---------- dump meta ----------
        with open(OUT_ROOT / "ablation_index.json", "w", encoding="utf-8") as f:
            json.dump(meta_all, f, indent=2, ensure_ascii=False)

        print(f"[DONE] All ablations finished. See: {OUT_ROOT}")

    finally:
        # ---------- restore original yamls ----------
        write_yaml(BASE_YAML, base0)
        write_yaml(SSL_YAML,  ssl0)
        write_yaml(FT_YAML,   ft0)
        write_yaml(PRIV_YAML, priv0)
        write_yaml(DYN_YAML,  dyn0)
        print("[INFO] Restored original configs.")

if __name__ == "__main__":
    main()
