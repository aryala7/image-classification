"""Log per-sample softmax probabilities for test & custom_test to SEPARATE runs.

AUROC and ECE need per-sample probabilities, which the training pipeline never
logged (only argmax). This script does NOT compute those metrics -- it only
*produces the raw material* so the eval branch can compute them from artifacts,
the same way it already derives acc/F1/MCC from the confusion CSVs.

For each selected source run it:
  1. takes ITS OWN logged model-checkpoint artifact (correct weights, guaranteed),
  2. rebuilds the architecture and loads those weights,
  3. re-runs inference over test + custom_test to get softmax probabilities,
  4. starts a new `<source>_<suffix>` run (e.g. baseline_final_softmax), links the
     source checkpoint as an input artifact (lineage), and logs a probabilities
     CSV artifact per split: columns [true_label, p_<class0>, ...].

No retraining, no metric computation here. Safe by default: computes + prints a
weight-correctness check; creates nothing unless --write.

Usage (from repo root):
    python -m scripts.log_prob_metrics                 # dry-run: infer + sanity print
    python -m scripts.log_prob_metrics --write         # create the _softmax runs
    python -m scripts.log_prob_metrics --limit 1 --write
"""
from __future__ import annotations

import argparse
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

# run.config["model"] stores model.name; map the ones that differ from their
# MODEL_REGISTRY key (resnet18's factory sets model.name = "baseline").
MODEL_ALIASES = {"baseline": "resnet18"}

SPLITS = ["test", "custom_test"]
PROB_DIR = Path("prob_data")


def resolve_factory_key(model_name: str) -> str:
    key = MODEL_ALIASES.get(model_name, model_name)
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Cannot map run model '{model_name}' to a MODEL_REGISTRY key "
            f"(tried '{key}'). Available: {sorted(MODEL_REGISTRY)}"
        )
    return key


def get_checkpoint(run):
    """Return ``(local_path, artifact)`` for the run's OWN model checkpoint."""
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        local = Path(art.download())
        files = sorted(local.glob("*.pth")) + sorted(local.glob("*.pt"))
        if files:
            return files[0], art
    raise FileNotFoundError(f"No 'model' artifact found for run {run.name!r}")


def build_model(run):
    """Rebuild the architecture and load the source run's trained weights."""
    model_name = run.config.get("model")
    num_classes = run.config.get("num_classes")
    key = resolve_factory_key(model_name)
    model = MODEL_REGISTRY[key](num_classes=num_classes)  # sets .transforms/.name
    ckpt_path, artifact = get_checkpoint(run)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    return model, num_classes, artifact


def build_loaders(model, num_classes):
    """test + custom_test loaders using the model's own eval transforms."""
    exclude = ["trash"] if num_classes == len(CLASS_NAMES) - 1 else None
    _, _, test_loader = create_training_dataloaders(
        transform=model.transforms,
        batch_size=BATCH_SIZE,
        data_config=data_config,
        exclude_classes=exclude,
    )
    custom_loader = create_custom_test_dataloader(
        transform=model.transforms,
        batch_size=BATCH_SIZE,
        data_config=data_config,
        exclude_classes=exclude,
    )
    return {"test": test_loader, "custom_test": custom_loader}


@torch.inference_mode()
def run_inference(model, data_loader) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` -- softmax probs (N, C) and true labels (N,)."""
    model.eval()
    prob_chunks, label_chunks = [], []
    for X, y in tqdm(data_loader, desc="inference", leave=False):
        logits = model(X.to(DEVICE))
        prob_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        label_chunks.append(np.asarray(y))
    return np.concatenate(prob_chunks), np.concatenate(label_chunks)


def probs_to_frame(probs: np.ndarray, labels: np.ndarray, class_names) -> pd.DataFrame:
    """Per-sample frame: true_label + one probability column per class."""
    frame = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
    frame.insert(0, "true_label", np.asarray(labels))
    return frame


def log_probs_artifact(new_run, split: str, frame: pd.DataFrame) -> None:
    PROB_DIR.mkdir(exist_ok=True)
    csv_path = PROB_DIR / f"{new_run.name}_{split}_probs.csv"
    frame.to_csv(csv_path, index=False)
    artifact = wandb.Artifact(name=f"{new_run.name}_{split}_probs", type="probabilities")
    artifact.add_file(str(csv_path))
    new_run.log_artifact(artifact)


def link_source_checkpoint(new_run, artifact) -> None:
    """Record lineage: this run used those weights (best-effort)."""
    if artifact is None:
        return
    last_exc = None
    for ref in (artifact, getattr(artifact, "qualified_name", None), getattr(artifact, "name", None)):
        if ref is None:
            continue
        try:
            new_run.use_artifact(ref)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    print(f"  (could not link source artifact: {last_exc})")


def process_source(source_run, args, entity, project) -> None:
    print(f"\n=== source: {source_run.name} (model={source_run.config.get('model')}, "
          f"classes={source_run.config.get('num_classes')}) ===")
    model, num_classes, artifact = build_model(source_run)
    loaders = build_loaders(model, num_classes)
    class_names = CLASS_NAMES[:num_classes]

    # Infer both splits first, so dry-run can print the weight-correctness check
    # without creating a run.
    frames, sanity = {}, {}
    for split in SPLITS:
        probs, labels = run_inference(model, loaders[split])
        frames[split] = probs_to_frame(probs, labels, class_names)
        acc = float((probs.argmax(axis=1) == labels).mean())
        sanity[split] = acc

        logged = source_run.summary.get(f"{split}/acc")
        note = ""
        if isinstance(logged, (int, float)):
            delta = abs(acc - logged)
            flag = "OK" if delta < 1e-3 else "MISMATCH!"
            note = f" | acc {acc:.4f} vs logged {logged:.4f} (Δ={delta:.4f}) [{flag}]"
        print(f"  {split:12} N={len(frames[split])}{note}")

    if not args.write:
        print(f"  [dry-run] would create '{source_run.name}_{args.suffix}' "
              f"with {len(SPLITS)} probability artifact(s).")
        return

    new_run = wandb.init(
        entity=entity,
        project=project,
        name=f"{source_run.name}_{args.suffix}",
        job_type="probability_logging",
        reinit=True,
        tags=list(source_run.tags) + [args.new_tag],
        config={
            "model": source_run.config.get("model"),
            "num_classes": num_classes,
            "dataset": source_run.config.get("dataset"),
            "epochs": source_run.config.get("epochs"),
            "source_run_name": source_run.name,
            "source_run_id": source_run.id,
        },
    )
    link_source_checkpoint(new_run, artifact)
    for split in SPLITS:
        log_probs_artifact(new_run, split, frames[split])
        new_run.summary[f"{split}/acc"] = sanity[split]  # cheap weight-sanity scalar
    new_run.finish()
    print(f"  created run '{new_run.name}' with {len(SPLITS)} probability artifact(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="core", help="source run tag to select (None-> all)")
    parser.add_argument("--suffix", default="softmax", help="appended to source run name")
    parser.add_argument("--new-tag", default="softmax", help="tag applied to the new runs")
    parser.add_argument("--write", action="store_true", help="create the new wandb runs")
    parser.add_argument("--limit", type=int, default=None, help="process only N runs (debug)")
    args = parser.parse_args()

    entity = training_config["wandb"]["entity"]
    project = training_config["wandb"]["project"]
    print(f"Device: {DEVICE} | project: {entity}/{project} | "
          f"mode: {'WRITE (new runs)' if args.write else 'dry-run'}")

    api = wandb.Api()
    filters = {"tags": {"$in": [args.tag]}} if args.tag and args.tag != "None" else None
    runs = list(api.runs(f"{entity}/{project}", filters=filters))
    runs = [r for r in runs if not r.name.endswith(f"_{args.suffix}")]  # skip our own outputs
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        raise RuntimeError("No source runs matched. Check auth, project, and --tag.")

    print(f"Processing {len(runs)} source run(s): {[r.name for r in runs]}")
    for run in runs:
        process_source(run, args, entity, project)

    print("\nDone." + ("" if args.write else " (dry-run -- rerun with --write to create runs)"))


if __name__ == "__main__":
    main()
