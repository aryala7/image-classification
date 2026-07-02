"""Log per-sample softmax probabilities for test & custom_test to separate runs.

Why this exists:
    Our training runs only saved the *final predicted class* for each image, but
    metrics like AUROC and ECE need the model's full *probability scores* (how
    confident it was in every class). This script goes back to each finished
    model, re-runs the test images through it once to recover those scores, and
    saves them to Weights & Biases so the eval branch can compute those metrics.

    Nothing is re-trained: we only reuse the weights that were already saved.

What it produces:
    For every run tagged "core", a new run named "<original>_softmax" containing,
    for both the normal test set and the custom (domain-shift) set, a CSV with one
    row per image: its true label plus the probability the model gave each class.

    python -m scripts.log_prob_metrics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm

from src.data import create_custom_test_dataloader, create_training_dataloaders
from src.engine import CLASS_NAMES
from src.models import MODEL_REGISTRY
from src.train import BATCH_SIZE, DEVICE, data_config, training_config

# --- settings ---------------------------------------------------------------
SOURCE_TAG = "core"        # only process runs carrying this tag
SUFFIX = "softmax"         # new runs are named "<original>_softmax"

MODEL_ALIASES = {"baseline": "resnet18"}
PROB_DIR = Path("prob_data")  # local folder for the CSVs before they're uploaded


def load_weights(run):
    """Rebuild the model architecture and load this run's trained weights."""
    # Pick the right model-building function based on what the run recorded.
    key = MODEL_ALIASES.get(run.config["model"], run.config["model"])
    model = MODEL_REGISTRY[key](num_classes=run.config["num_classes"])

    # Each run saved its own weights file ("model" artifact) - grab that one so we
    # can never mix up which weights belong to which run.
    art = next((a for a in run.logged_artifacts() if a.type == "model"), None)
    if art is None:
        raise FileNotFoundError(f"No 'model' artifact for run {run.name!r}")
    ckpt = sorted(Path(art.download()).glob("*.pth")) + sorted(Path(art.download()).glob("*.pt"))
    model.load_state_dict(torch.load(ckpt[0], map_location=DEVICE))
    return model.to(DEVICE)


def split_loaders(model, num_classes):
    """Build the two evaluation sets: the normal test set and the custom set."""
    # If this run was trained without the "trash" class, leave it out here too.
    exclude = ["trash"] if num_classes == len(CLASS_NAMES) - 1 else None
    # create_training_dataloaders returns train/val/test; we only need test.
    *_, test = create_training_dataloaders(model.transforms, BATCH_SIZE, data_config, exclude_classes=exclude)
    custom = create_custom_test_dataloader(model.transforms, BATCH_SIZE, data_config, exclude_classes=exclude)
    return {"test": test, "custom_test": custom}


@torch.inference_mode()
def infer(model, loader):
    """Run every image through the model and collect probability scores + labels."""
    model.eval()
    probs, labels = [], []
    for X, y in tqdm(loader, desc="inference", leave=False):
        # softmax turns the raw model outputs into probabilities that sum to 1.
        probs.append(torch.softmax(model(X.to(DEVICE)), dim=1).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def main():
    entity, project = training_config["wandb"]["entity"], training_config["wandb"]["project"]
    PROB_DIR.mkdir(exist_ok=True)

    # Find every run we want to process (the "core" ones).
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"tags": {"$in": [SOURCE_TAG]}}))
    print(f"{DEVICE} | {len(runs)} '{SOURCE_TAG}' run(s): {[r.name for r in runs]}")

    for run in runs:
        num_classes = run.config["num_classes"]
        class_names = CLASS_NAMES[:num_classes]
        model = load_weights(run)
        loaders = split_loaders(model, num_classes)

        # Start a fresh run for the probabilities so the original run stays untouched.
        # "source_run_name/id" let the eval branch tie these back to the original.
        new_run = wandb.init(
            entity=entity, project=project, name=f"{run.name}_{SUFFIX}",
            job_type="probability_logging", reinit=True, tags=[SUFFIX],
            config={"model": run.config["model"], "num_classes": num_classes,
                    "source_run_name": run.name, "source_run_id": run.id},
        )
        for split, loader in loaders.items():
            probs, labels = infer(model, loader)
            # One row per image: the true label + one probability column per class.
            frame = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
            frame.insert(0, "true_label", labels)
            # Save locally, then upload as a wandb artifact the eval branch can read.
            csv = PROB_DIR / f"{new_run.name}_{split}_probs.csv"
            frame.to_csv(csv, index=False)
            artifact = wandb.Artifact(f"{new_run.name}_{split}_probs", type="probabilities")
            artifact.add_file(str(csv))
            new_run.log_artifact(artifact)
        new_run.finish()
        print(f"  logged {new_run.name}")


if __name__ == "__main__":
    main()
