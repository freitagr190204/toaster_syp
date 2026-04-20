"""Shared toast classifier + ImageNet preprocessing (train + live must match)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """x: Nx3xHxW float in [0, 1]."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_toast_classifier(num_classes: int, pretrained_backbone: bool = True) -> nn.Module:
    """ResNet-18 head for 5-way browning; pretrained backbone improves real lighting / texture."""
    try:
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        m = models.resnet18(weights=w)
    except (AttributeError, TypeError):
        m = models.resnet18(pretrained=pretrained_backbone)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m
