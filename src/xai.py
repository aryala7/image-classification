"""XAI utilities for waste classifiers."""

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn
from torch.nn import functional as F
from captum.attr import Occlusion as CaptumOcclusion


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last Conv2d layer in a model."""
    last_conv_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module

    if last_conv_layer is None:
        raise ValueError("Grad-CAM needs a model with at least one Conv2d layer.")

    return last_conv_layer


class GradCAM:
    """Compute Grad-CAM heatmaps for a model and target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, int, float]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = predicted_class

        target_score = logits[:, target_class].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # Multiply each feature map by its importance weight.
        # Sum the weighted maps.
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # Resize the heatmap to the original image size.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        confidence = float(probabilities[:, target_class].item())
        return cam, predicted_class, target_class, confidence


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on a PIL image."""
    image = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        image.size,
        resample=Image.BILINEAR,
    )
    heatmap_array = np.asarray(heatmap_image) / 255.0
    colored_heatmap = colormaps["jet"](heatmap_array)[..., :3]

    image_array = np.asarray(image).astype(np.float32) / 255.0
    overlay = (1 - alpha) * image_array + alpha * colored_heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_grad_cam(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image=image, heatmap=heatmap, alpha=alpha)
    overlay.save(output_path)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap


def attribution_to_heatmap(
    attribution: torch.Tensor,
    positive_only: bool = True,
) -> np.ndarray:
    """Convert a 3-channel attribution tensor into a normalized 2D heatmap."""
    if attribution.ndim == 4:
        attribution = attribution.squeeze(0)

    if positive_only:
        attribution = attribution.clamp(min=0)
    else:
        attribution = attribution.abs()

    heatmap = attribution.detach().mean(dim=0).cpu().numpy()
    return _normalize_heatmap(heatmap)


class OcclusionAttribution:
    """Compute Captum patch-occlusion attributions for any image classifier."""

    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 32,
        stride: int = 16,
        baseline: float = 0.0,
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.baseline = baseline
        self._captum_occlusion = CaptumOcclusion(model)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, int, float]:
        self.model.eval()

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = predicted_class

        attribution = self._captum_occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=(
                input_tensor.shape[1],
                self.patch_size,
                self.patch_size,
            ),
            strides=(input_tensor.shape[1], self.stride, self.stride),
            baselines=self.baseline,
        )
        heatmap = attribution_to_heatmap(attribution, positive_only=True)

        confidence = float(probabilities[:, target_class].item())
        return heatmap, predicted_class, target_class, confidence


def save_occlusion(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image=image, heatmap=heatmap, alpha=alpha)
    overlay.save(output_path)
