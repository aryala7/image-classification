"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import yaml
import wandb
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml"):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               log_artifact: bool = False):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    if log_artifact:
        artifact = wandb.Artifact(name=model_name, type="model")
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)