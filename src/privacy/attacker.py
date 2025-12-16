import torch
import torch.nn as nn


class FeatureAttacker(nn.Module):
    """
    Auxiliary classifier for FLR estimation
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)
