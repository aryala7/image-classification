"""Generate Integrated Gradients overlays for saved model checkpoints."""

import json
from pathlib import Path

import torch
from datasets import load_dataset

from src.data import (
    CUSTOM_TEST_REPO_ID,
    CUSTOM_TEST_SPLIT,
    prepare_rgb_image,
)
from src.models import MODEL_REGISTRY
from src.xai import IntegratedGradientsAttribution, save_integrated_gradients

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = list(MODEL_REGISTRY.keys())
CHECKPOINT_PATH = None
DATASET_REPO_ID = CUSTOM_TEST_REPO_ID
DATASET_SPLIT = CUSTOM_TEST_SPLIT
OUTPUT_DIR = "xai_outputs/integrated_gradients"
NUM_IMAGES = 8
SAMPLE_MODE = "balanced"  # "balanced" or "first"
TARGET_CLASS_MODE = "predicted"  # "predicted" or "true"
BASELINE = 0.0
N_STEPS = 50
INTERNAL_BATCH_SIZE = None
ATTRIBUTION_MODE = "absolute"  # "absolute" or "positive"
DEFAULT_CHECKPOINTS = {
    "resnet18": "models/baseline_5epochs.pth",
    "mlp_mixer": "models/mlp_mixer_3epochs.pth",
    "convnext_tiny": "models/convnext_tiny_3epochs.pth",
    "deit": "models/deit_3epochs.pth",
}
CHECKPOINT_STEMS = {
    "resnet18": ("baseline", "resnet18"),
    "mlp_mixer": ("mlp_mixer",),
    "convnext_tiny": ("convnext_tiny",),
    "deit": ("deit",),
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_class_names() -> tuple[int, list[str]]:
    final_classes_path = PROJECT_ROOT / "configs" / "final_classes.json"
    with final_classes_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return len(class_names), class_names


def select_examples(dataset, num_classes: int, num_images: int, sample_mode: str):
    if sample_mode == "first":
        return dataset.select(range(min(num_images, len(dataset))))

    examples = []
    class_counts = {class_id: 0 for class_id in range(num_classes)}
    max_per_class = max(1, (num_images + num_classes - 1) // num_classes)

    for item in dataset:
        label = int(item["label"])
        if class_counts[label] >= max_per_class:
            continue

        examples.append(item)
        class_counts[label] += 1

        if len(examples) >= num_images:
            break

    return examples


def resolve_checkpoint_path(
    model_key: str,
    explicit_checkpoint: str | None,
) -> Path | None:
    if explicit_checkpoint is not None:
        return PROJECT_ROOT / explicit_checkpoint

    configured = PROJECT_ROOT / DEFAULT_CHECKPOINTS[model_key]
    if configured.exists():
        return configured

    for model_stem in CHECKPOINT_STEMS[model_key]:
        candidates = sorted(
            (PROJECT_ROOT / "models").glob(f"{model_stem}_*epochs.pth")
        )
        if candidates:
            return candidates[-1]

    return None


def load_model(model_key: str, num_classes: int, checkpoint_path: Path):
    model = MODEL_REGISTRY[model_key](num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def generate_for_model(
    model_key: str,
    checkpoint_path: Path,
    dataset,
    class_names: list[str],
) -> None:
    model = load_model(
        model_key=model_key,
        num_classes=len(class_names),
        checkpoint_path=checkpoint_path,
    )
    transform = model.transforms
    integrated_gradients = IntegratedGradientsAttribution(
        model=model,
        baseline=BASELINE,
        n_steps=N_STEPS,
        internal_batch_size=INTERNAL_BATCH_SIZE,
        attribution_mode=ATTRIBUTION_MODE,
    )

    print(f"Model: {model_key}")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print("Integrated Gradients backend: captum")

    examples = select_examples(
        dataset=dataset,
        num_classes=len(class_names),
        num_images=NUM_IMAGES,
        sample_mode=SAMPLE_MODE,
    )

    for idx, item in enumerate(examples):
        image = prepare_rgb_image(item["image"])
        label = int(item["label"])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        target_class = label if TARGET_CLASS_MODE == "true" else None
        (
            _heatmap,
            attribution_rgb,
            predicted_label,
            integrated_label,
            confidence,
        ) = integrated_gradients(input_tensor, target_class=target_class)
        output_path = Path(
            f"{PROJECT_ROOT / OUTPUT_DIR}/"
            f"{model_key}_{DATASET_SPLIT}_{idx:03d}_"
            f"integrated-gradients-{TARGET_CLASS_MODE}_"
            f"true-{class_names[label]}_"
            f"pred-{class_names[predicted_label]}_"
            f"target-{class_names[integrated_label]}_"
            f"conf-{confidence:.2f}.jpg"
        )

        save_integrated_gradients(
            image=image,
            attribution_rgb=attribution_rgb,
            output_path=output_path,
            left_title=class_names[integrated_label],
        )
        print(output_path)


def main() -> None:
    if CHECKPOINT_PATH is not None and len(MODEL_NAMES) != 1:
        raise ValueError(
            "CHECKPOINT_PATH can only be used with exactly one MODEL_NAMES value."
        )

    num_classes, class_names = load_class_names()
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATASET_REPO_ID} [{DATASET_SPLIT}]")
    dataset = load_dataset(DATASET_REPO_ID, split=DATASET_SPLIT)

    for model_key in MODEL_NAMES:
        checkpoint_path = resolve_checkpoint_path(model_key, CHECKPOINT_PATH)
        if checkpoint_path is None or not checkpoint_path.exists():
            print(f"Skipping {model_key}: no checkpoint found.")
            continue

        generate_for_model(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            class_names=class_names,
        )
        print("-" * 50)


if __name__ == "__main__":
    main()
