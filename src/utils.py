"""
Contains various utility functions for PyTorch model training and saving.
"""

import os
import random

import numpy as np
import torch
import yaml
import wandb
from pathlib import Path
from typing import Callable, Optional


def seed_everything(seed=42):
    # 1. Python & Environment
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # 2. PyTorch CPU
    torch.manual_seed(seed)

    # 3. PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # 4. CuDNN Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "configs/config.yaml"):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_model(
    model: torch.nn.Module, target_dir: str, model_name: str, log_artifact: bool = False
):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    if log_artifact:
        artifact = wandb.Artifact(name=model_name, type="model")
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)


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


def load_model_from_artifact(
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
