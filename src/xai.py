"""XAI utilities for waste classifiers."""

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn
from torch.nn import functional as F
from captum.attr import IntegratedGradients as CaptumIntegratedGradients
from captum.attr import Occlusion as CaptumOcclusion


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the target layer for Grad-CAM.

    For ConvNeXt models, use the last block in the final stage.
    Otherwise fall back to the last Conv2d layer.
    """
    if hasattr(model, "stages"):
        try:
            last_block = model.stages[-1].blocks[-1]
            if isinstance(last_block, nn.Module):
                return last_block
        except Exception:
            pass

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


def overlay_attribution_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.75,
) -> Image.Image:
    """Overlay an attribution heatmap without tinting low-attribution regions."""
    image = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        image.size,
        resample=Image.BILINEAR,
    )
    heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0
    colored_heatmap = colormaps["jet"](heatmap_array)[..., :3]

    image_array = np.asarray(image).astype(np.float32) / 255.0
    alpha_mask = (alpha * heatmap_array)[..., None]
    overlay = (1 - alpha_mask) * image_array + alpha_mask * colored_heatmap
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


def _normalize_heatmap(
    heatmap: np.ndarray,
    percentile: float | None = 99.0,
) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    if percentile is None:
        max_value = heatmap.max()
    else:
        max_value = np.percentile(heatmap, percentile)

    if max_value > 0:
        heatmap = np.clip(heatmap / max_value, 0, 1)

    return heatmap


def attribution_to_heatmap(
    attribution: torch.Tensor,
    mode: str = "positive",
    percentile: float | None = 99.0,
) -> np.ndarray:
    """Convert a 3-channel attribution tensor into a normalized 2D heatmap."""
    if attribution.ndim == 4:
        attribution = attribution.squeeze(0)

    if mode == "positive":
        attribution = attribution.clamp(min=0)
    elif mode == "absolute":
        attribution = attribution.abs()
    else:
        raise ValueError(f"Unsupported attribution heatmap mode: {mode}")

    heatmap = attribution.detach().mean(dim=0).cpu().numpy()
    return _normalize_heatmap(heatmap, percentile=percentile)


def attribution_to_rgb(
    attribution: torch.Tensor,
    mode: str = "signed",
    percentile: float | None = 99.0,
) -> np.ndarray:
    """Convert a 3-channel attribution tensor into an RGB attribution image."""
    if attribution.ndim == 4:
        attribution = attribution.squeeze(0)

    if mode == "positive":
        attribution = attribution.clamp(min=0)
    elif mode == "absolute":
        attribution = attribution.abs()
    elif mode != "signed":
        raise ValueError(f"Unsupported attribution RGB mode: {mode}")

    attribution_array = attribution.detach().cpu().numpy().transpose(1, 2, 0)
    if mode == "signed":
        max_value = np.percentile(np.abs(attribution_array), percentile or 100)
        if max_value > 0:
            attribution_array = np.clip(attribution_array / max_value, -1, 1)
        attribution_array = (attribution_array + 1) / 2
    else:
        attribution_array = attribution_array - attribution_array.min()
        if percentile is None:
            max_value = attribution_array.max()
        else:
            max_value = np.percentile(attribution_array, percentile)
        if max_value > 0:
            attribution_array = np.clip(attribution_array / max_value, 0, 1)

    return attribution_array.astype(np.float32)


def save_side_by_side_attribution(
    image: Image.Image,
    attribution_rgb: np.ndarray,
    output_path: Path,
    left_title: str,
    right_title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.convert("RGB")
    attribution_image = Image.fromarray(np.uint8(attribution_rgb * 255)).resize(
        image.size,
        resample=Image.BILINEAR,
    )

    title_height = max(28, image.height // 24)
    gap = max(8, image.width // 80)
    panel_width = image.width * 2 + gap
    panel_height = image.height + title_height

    panel = Image.new("RGB", (panel_width, panel_height), "white")
    panel.paste(image, (0, title_height))
    panel.paste(attribution_image, (image.width + gap, title_height))

    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        for title, x, width in (
            (left_title, 0, image.width),
            (right_title, image.width + gap, image.width),
        ):
            text_box = draw.textbbox((0, 0), title, font=font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x + (width - text_width) // 2, 8),
                title,
                fill="black",
                font=font,
            )
    except Exception:
        pass

    panel.save(output_path)


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
        heatmap = attribution_to_heatmap(attribution, mode="positive")

        confidence = float(probabilities[:, target_class].item())
        return heatmap, predicted_class, target_class, confidence


def save_occlusion(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.75,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_attribution_heatmap(image=image, heatmap=heatmap, alpha=alpha)
    overlay.save(output_path)


class IntegratedGradientsAttribution:
    """Compute Captum Integrated Gradients attributions for any image classifier."""

    def __init__(
        self,
        model: nn.Module,
        baseline: float = 0.0,
        n_steps: int = 50,
        internal_batch_size: int | None = None,
        attribution_mode: str = "absolute",
    ):
        self.model = model
        self.baseline = baseline
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.attribution_mode = attribution_mode
        self._integrated_gradients = CaptumIntegratedGradients(model)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, int, float]:
        self.model.eval()

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = predicted_class

        attribution = self._integrated_gradients.attribute(
            input_tensor,
            target=target_class,
            baselines=self.baseline,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
        )
        heatmap = attribution_to_heatmap(attribution, mode=self.attribution_mode)
        attribution_rgb = attribution_to_rgb(attribution, mode=self.attribution_mode)

        confidence = float(probabilities[:, target_class].item())
        return heatmap, attribution_rgb, predicted_class, target_class, confidence


def save_integrated_gradients(
    image: Image.Image,
    attribution_rgb: np.ndarray,
    output_path: Path,
    left_title: str,
    right_title: str = "Integrated Gradient",
) -> None:
    save_side_by_side_attribution(
        image=image,
        attribution_rgb=attribution_rgb,
        output_path=output_path,
        left_title=left_title,
        right_title=right_title,
    )
