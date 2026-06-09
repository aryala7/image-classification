from models.convnext import build_convnext_tiny
from models.deit import build_deit_tiny
from models.mlp_mixer import build_mlp_mixer


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True
):
    if model_name == "convnext_tiny":
        return build_convnext_tiny(num_classes, pretrained)

    elif model_name == "deit_tiny":
        return build_deit_tiny(num_classes, pretrained)

    elif model_name == "mlp_mixer":
        return build_mlp_mixer(num_classes, pretrained)

    else:
        raise ValueError(f"Unknown model name: {model_name}")