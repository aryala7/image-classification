import timm
import torch.nn as nn


def build_mlp_mixer(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "mixer_b16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model