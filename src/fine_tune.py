import json
from xml.parsers.expat import model

import torch
import wandb

from typing import Callable, Optional
from pathlib import Path

from src.models import MODEL_REGISTRY
from src.train import DEVICE


def _resolve_checkpoint_path(
    artifact_path: str,
    download_dir: str = "models",
) -> Path:
    """Resolve a checkpoint path from a local file or a W&B artifact reference."""
    artifact_name = artifact_path.split(":")[0].split("/")[-1]
    candidate = Path(download_dir) / artifact_name
    if candidate.exists() and candidate.is_file():
        return candidate

    download_dir_path = Path(download_dir)
    download_dir_path.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    artifact = api.artifact(artifact_path)

    # Otherwise download into the cache folder (artifact.download already creates a folder)
    downloaded_path = Path(artifact.download(root=str(download_dir_path)))
    return downloaded_path / artifact_name


def _load_model_from_artifact(
    model_fn: Callable[..., torch.nn.Module],
    artifact_path: str,
    num_classes: int = 6,
    artifact_file_name: Optional[str] = None,
    download_dir: str = "models",
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load a fine-tuned PyTorch model from a local checkpoint or a W&B artifact.

    Args:
        model_fn: Callable that returns a model instance when called with num_classes.
        num_classes: Number of output classes for the model.
        artifact_path: Local checkpoint path or W&B artifact reference
            (e.g. "entity/project/artifact_name:version").
        artifact_file_name: Optional checkpoint file name inside the artifact.
        download_dir: Directory to store downloaded W&B artifact files.
        device: Optional device for loading and returning the model.

    Returns:
        A model with the loaded weights.
    """
    device = device or torch.device("cpu")
    checkpoint_path = _resolve_checkpoint_path(
        artifact_path=artifact_path,
        download_dir=download_dir,
    )

    if checkpoint_path.absolute().suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    model = model_fn(num_classes=num_classes)
    model.load_state_dict(state_dict)

    model.to(device)
    return model


def get_fine_tuned_models():
    resnet18_20 = _load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("resnet18"),
        artifact_path="trash-busters/baseline_20epochs.pth:latest",
        device=DEVICE
    )

    mlp_mixer_20 = _load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("mlp_mixer"),
        artifact_path="trash-busters/mlp_mixer_20epochs.pth:latest",
        device=DEVICE
    )
    convnext_20 = _load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("convnext_tiny"),
        artifact_path="trash-busters/convnext_tiny_20epochs.pth:latest",
        device=DEVICE
    )
    deit_20 = _load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("deit"),
        artifact_path="trash-busters/deit_20epochs.pth:latest",
        device=DEVICE
    )

    convnext_no_trash = _load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("convnext_tiny"),
        artifact_path="trash-busters/convnext_tiny_no_trash_20epochs.pth:v0",
        device=DEVICE,
        num_classes=5
    )

    return {
        "resnet18_20": resnet18_20,
        "mlp_mixer_20": mlp_mixer_20,
        "convnext_20": convnext_20,
        "deit_20": deit_20,
        "convnext_no_trash": convnext_no_trash,
    }

def load_class_names() -> tuple[int, list[str]]:
    final_classes_path = Path(__file__).parent.parent / "configs" / "final_classes.json"
    with final_classes_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return len(class_names), class_names

def test_fine_tuned_model(model_name: str, image_path: str) -> tuple[int, str]:
    fine_tuned_models = get_fine_tuned_models()
    model = fine_tuned_models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found among fine-tuned models.")

    from PIL import Image
    image = Image.open(image_path)

    # Apply the model's transforms to the image
    transformed_image = model.transforms(image).unsqueeze(0).to(DEVICE)

    # Set the model to evaluation mode and perform inference
    model.eval()
    with torch.no_grad():
        output = model(transformed_image)
        predicted_class = output.argmax(dim=1).item()
    class_names = load_class_names()[1]

    # create gradcam heatmap if model is convnext or resnet18
    if model_name in ["convnext_20", "resnet18_20"]:
        from src.xai import find_last_conv_layer
        target_layer = find_last_conv_layer(model)
        _output_gradcam(
            model=model,
            image_path=image_path,
            target_layer=target_layer
        )
    return predicted_class, class_names[predicted_class]

def _output_gradcam(
    model: torch.nn.Module,
    image_path: str,
    target_layer: str,
    output_dir: str = "gradcam_outputs",
) -> None:
    from src.xai import GradCAM, save_grad_cam
    from PIL import Image

    # Load and preprocess the image
    image = Image.open(image_path)
    transformed_image = model.transforms(image).unsqueeze(0).to(DEVICE)

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Generate Grad-CAM heatmap
    # heatmap = grad_cam.generate(transformed_image)
    heatmap, predicted_label, cam_label, confidence = grad_cam(
        transformed_image,
    )

    # Save the Grad-CAM heatmap
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_location = output_path / f"gradcam_{Path(image_path).stem}.png"
    save_grad_cam(
        image=image,
        heatmap=heatmap,
        output_path=output_location,
    )
    print(f"Grad-CAM heatmap saved to: {output_location}")
