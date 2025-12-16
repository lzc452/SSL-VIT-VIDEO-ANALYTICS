import torch
import torch.nn as nn
import torch.nn.functional as F


def client_update(model, loader, device, epochs=1, lr=3e-4, weight_decay=0.01, amp=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    for _ in range(int(epochs)):
        for clip, y in loader:
            clip = clip.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(clip)
                loss = ce(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total += bs

    avg_loss = total_loss / max(1, total)
    return avg_loss
