import torch
import torch.nn.functional as F


def prediction_entropy(logits):
    prob = F.softmax(logits, dim=1)
    ent = -(prob * (prob + 1e-12).log()).sum(dim=1)
    return ent.mean().item()


def privacy_exposure_rate(before, after):
    return after / max(before, 1e-6)


def top1_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()
