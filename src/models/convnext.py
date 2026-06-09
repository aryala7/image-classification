import torch.nn as nn
import timm
from timm.data import resolve_model_data_config, create_transform
from torchinfo import summary



def build_convnext_tiny(
    num_classes: int,
    pretrained: bool = True,
    freeze_features: bool = True,
):
    """
    Build ConvNeXt-Tiny with timm for feature extraction fine-tuning.

    Returns:
        model: ConvNeXt-Tiny model with updated classifier.
        preprocess: Default validation/test transforms for the pretrained weights.
    """

    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes
    )

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.get_classifier().parameters():
            param.requires_grad = True

    data_config = resolve_model_data_config(model)
    preprocess = create_transform(
        **data_config,
        is_training=False
    )

    model.name = "convnext_tiny"
    summary(model=model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )

    return model, preprocess


if __name__ == "__main__":
    build_convnext_tiny(6)