from __future__ import annotations

import argparse
import torch

from src.mae.utils import load_yaml, set_seed
from src.datasets.mae_dataset import MAEVideoDataset  # 你现在发来的文件名是 mae_dataset.py
from src.models.tinyvit_mae import TinyViTMAE
from src.mae.masking import make_token_mask, count_masked, count_visible
from src.mae.losses import reconstruction_error_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # 1) dataset: 取一个 clip
    # -----------------------
    ds_cfg = cfg["dataset"]
    inp = cfg.get("input", {})

    dataset = MAEVideoDataset(
        split_path=str(ds_cfg["train_split"]),
        image_size=int(inp.get("image_size", ds_cfg.get("image_size", 112))),
        clip_len=int(inp.get("clip_len", ds_cfg.get("clip_len", 32))),
        stride=int(inp.get("stride", ds_cfg.get("stride", 4))),
        mean=tuple(inp.get("mean", ds_cfg.get("mean", [0.485, 0.456, 0.406]))),
        std=tuple(inp.get("std", ds_cfg.get("std", [0.229, 0.224, 0.225]))),
        training=False,
    )

    clip = dataset[0].unsqueeze(0).to(device)  # [1,C,T,H,W]
    print("[SANITY] clip:", tuple(clip.shape), clip.dtype, clip.device)

    # -----------------------
    # 2) model: 严格按 train_mae.py 的关键字参数构造
    # -----------------------
    m = cfg["model"]
    mae = cfg["mae"]

    stage4_pool = int(mae["stage4_pool"])
    model = TinyViTMAE(
        backbone_name=str(m["backbone"]),
        pretrained=str(m.get("pretrained", "")),
        stage4_pool=stage4_pool,
        decoder_dim=int(m["decoder_dim"]),
        decoder_depth=int(m["decoder_depth"]),
        decoder_num_heads=int(m["decoder_num_heads"]),
        mlp_ratio=float(m["decoder_mlp_ratio"]),
        drop_rate=float(m.get("drop_rate", 0.0)),
        attn_drop_rate=float(m.get("attn_drop_rate", 0.0)),
    ).to(device)

    model.train()

    # -----------------------
    # 3) mask: 严格按 masking.py 的定义 N=T*tokens_per_frame
    # -----------------------
    B, C, T, H, W = clip.shape
    tokens_per_frame = stage4_pool * stage4_pool  # 与 TinyViTMAE.tokens_per_frame 一致
    token_mask = make_token_mask(
        B=B,
        T=T,
        tokens_per_frame=tokens_per_frame,
        mask_ratio=float(mae["mask_ratio"]),
        mode=str(mae["mask_mode"]),
        device=device,
    )
    print("[SANITY] token_mask:", tuple(token_mask.shape), token_mask.dtype, "masked=", count_masked(token_mask), "visible=", count_visible(token_mask))

    # -----------------------
    # 4) forward + stats + backward
    # -----------------------
    amp_enabled = bool(cfg.get("training", {}).get("amp", True) and device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        pred, target = model(clip, token_mask=token_mask)

    print("[SANITY] pred:", tuple(pred.shape))
    print("[SANITY] target:", tuple(target.shape))

    stats = reconstruction_error_stats(pred.detach(), target.detach())
    print("[SANITY] stats:", stats)

    loss = (pred - target).pow(2).mean()
    loss.backward()

    print("[SANITY] backward OK")
    print("[SANITY] PASSED")


if __name__ == "__main__":
    main()
