import torchvision
from torch import nn


def resnet18(num_classes):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    model.name = "baseline"
    model.transforms = weights.transforms()
    return model


