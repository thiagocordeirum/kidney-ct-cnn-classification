import torch.nn as nn
from torchvision import models


def build():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m
