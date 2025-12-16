import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config
from datasets.loader import LazyFrameDataset
from models.mobilevit import build_mobilevit_s
from privacy.visual_mask import YuNetFaceDetector, VisualAnonymizer
from privacy.feature_noise import add_gaussian_noise, apply_feature_mask
from privacy.metrics_privacy import prediction_entropy, privacy_exposure_rate, top1_accuracy
from privacy.attacker import FeatureAttacker


def main():
    base = load_config("configs/base.yaml")
    cfg = load_config("configs/privacy.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    split = Path(base["paths"]["split_root"]) / cfg["dataset"]["split"]
    dataset = LazyFrameDataset(
        split_file=str(split),
        mode="supervised",
        clip_len=base["dataset"]["clip_len"],
        stride=base["dataset"]["stride"],
        image_size=base["dataset"]["image_size"]
    )
    loader = DataLoader(dataset, batch_size=cfg["runtime"]["batch_size"], shuffle=False)

    # Model
    backbone = build_mobilevit_s(embed_dim=cfg["model"]["embed_dim"])
    classifier = torch.nn.Linear(cfg["model"]["embed_dim"], cfg["dataset"]["num_classes"])
    backbone.load_state_dict(torch.load(cfg["model"]["finetune_ckpt"], map_location="cpu"), strict=False)
    backbone.to(device).eval()
    classifier.to(device).eval()

    # Feature privacy
    rows = ["sigma,mask_ratio,acc,entropy,flr\n"]
    for sigma in cfg["feature_privacy"]["noise_sigmas"]:
        for mask_ratio in cfg["feature_privacy"]["mask_ratios"]:
            zs, ys = [], []
            for clip, label in loader:
                clip, label = clip.to(device), label.to(device)
                with torch.no_grad():
                    feats = []
                    for t in range(clip.shape[2]):
                        _, z = backbone(clip[:, :, t])
                        feats.append(z)
                    z = torch.stack(feats, 1).mean(1)

                z = add_gaussian_noise(z, sigma)
                z = apply_feature_mask(z, mask_ratio)

                logits = classifier(z)
                zs.append(z.cpu())
                ys.append(label.cpu())

            zs = torch.cat(zs)
            ys = torch.cat(ys)

            acc = top1_accuracy(classifier(zs.to(device)), ys.to(device))
            ent = prediction_entropy(classifier(zs.to(device)))

            attacker = FeatureAttacker(zs.shape[1], cfg["dataset"]["num_classes"]).to(device)
            opt = torch.optim.Adam(attacker.parameters(), lr=cfg["feature_privacy"]["attacker_lr"])

            for _ in range(cfg["feature_privacy"]["attacker_epochs"]):
                opt.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(attacker(zs.to(device)), ys.to(device))
                loss.backward()
                opt.step()

            flr = top1_accuracy(attacker(zs.to(device)), ys.to(device))

            rows.append(f"{sigma},{mask_ratio},{acc:.4f},{ent:.4f},{flr:.4f}\n")
            print(f"[INFO] sigma={sigma}, mask={mask_ratio}, acc={acc:.3f}, flr={flr:.3f}")

    (save_dir / "feature_privacy.csv").write_text("".join(rows))
    print("[INFO] Privacy evaluation finished")


if __name__ == "__main__":
    main()
