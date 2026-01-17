import torch
from src.models.tinyvit_mae import TinyViTMAE
from src.mae.masking import make_token_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, T, H, W = 2, 3, 32, 112, 112
    model = TinyViTMAE(image_size=112, clip_len=32, embed_dim=256, stage4_pool=3).to(device)

    clip = torch.randn(B, C, T, H, W, device=device)
    N = T * (3 * 3)
    mask = make_token_mask(B, N, 0.9, "tube", T, 9, device)

    pred, tgt = model(clip, mask)
    assert pred.shape == tgt.shape == (B, N, 256), (pred.shape, tgt.shape)
    loss = ((pred[mask] - tgt[mask]) ** 2).mean()
    loss.backward()

    print("[OK] forward/backward, shapes:", pred.shape, "loss:", float(loss.item()))


if __name__ == "__main__":
    main()
