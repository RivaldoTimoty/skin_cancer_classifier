"""
model.py — EfficientNet-B0 Transfer Learning model with custom classification head.
Supports freeze/unfreeze strategies for progressive fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


class SkinCancerModel(nn.Module):
    """
    EfficientNet-B0 with custom classification head for 9-class skin cancer.
    
    Architecture:
        EfficientNet-B3 backbone (pretrained on ImageNet)
        → AdaptiveAvgPool2d
        → Dropout(0.3)
        → Linear(1280, 512)
        → BatchNorm1d(512)
        → ReLU
        → Dropout(0.4)
        → Linear(512, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B3
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)

        # Get the number of features from the backbone's classifier
        in_features = self.backbone.classifier[1].in_features  # 1536 for B3

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE_HEAD),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[MODEL] Backbone frozen")

    def unfreeze_backbone(self, num_layers: int = config.UNFREEZE_LAYERS) -> None:
        """Unfreeze the last `num_layers` layers of the backbone for fine-tuning."""
        params = list(self.backbone.parameters())
        total = len(params)
        for i, param in enumerate(params):
            if i >= total - num_layers:
                param.requires_grad = True
        trainable = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"[MODEL] Unfroze last {num_layers} layers ({trainable}/{total} trainable)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def get_model(pretrained: bool = True) -> SkinCancerModel:
    """Factory function: create and move model to the configured device."""
    model = SkinCancerModel(pretrained=pretrained)
    model = model.to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,}")
    print(f"[MODEL] Trainable params: {trainable_params:,}")
    return model
