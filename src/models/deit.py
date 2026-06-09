import timm
import torch.nn as nn


def build_deit_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model