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
