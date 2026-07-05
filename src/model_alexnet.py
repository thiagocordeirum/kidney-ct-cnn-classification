import torch.nn as nn
from torchvision import models


def build():
    m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[6] = nn.Linear(4096, 2)
    return m
