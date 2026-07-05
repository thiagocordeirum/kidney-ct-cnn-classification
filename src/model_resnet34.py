import torch.nn as nn
from torchvision import models


def build():
    m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m
