import torch.nn as nn
from torchvision import models


def build():
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[6] = nn.Linear(4096, 2)
    return m
