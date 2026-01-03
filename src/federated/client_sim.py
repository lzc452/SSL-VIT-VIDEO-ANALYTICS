import torch
import torch.nn as nn


def _get_amp_context(device, amp: bool):
    """
    Compatible autocast across torch versions.
    """
    enabled = bool(amp) and (device.type == "cuda")
    try:
        # torch >= 2.0 recommended
        autocast = torch.amp.autocast
        return autocast(device_type="cuda", enabled=enabled)
    except Exception:
        # fallback
        return torch.cuda.amp.autocast(enabled=enabled)


def _get_scaler(device, amp: bool):
    enabled = bool(amp) and (device.type == "cuda")
    if not enabled:
        return None
    try:
        # torch >= 2.0 recommended
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def client_update(model, loader, device, epochs=1, lr=3e-4, weight_decay=0.01, amp=True):
    """
    One client local training step.
    Returns: avg_loss (float)
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    ce = nn.CrossEntropyLoss()

    scaler = _get_scaler(device, amp)

    total_loss = 0.0
    total = 0

    for _ in range(int(epochs)):
        for clip, y in loader:
            clip = clip.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with _get_amp_context(device, amp):
                logits = model(clip)
                loss = ce(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = int(y.size(0))
            total_loss += float(loss.item()) * bs
            total += bs

    return total_loss / max(1, total)
