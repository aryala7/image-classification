"""Generate Grad-CAM overlays for saved model checkpoints."""

import json
from pathlib import Path

import torch
from datasets import load_dataset

from src.data import (
    CUSTOM_TEST_REPO_ID,
    CUSTOM_TEST_SPLIT,
    HF_REPO_ID,
    TEST_SPLIT,
    prepare_rgb_image,
)
from src.xai import GradCAM, find_last_conv_layer, save_grad_cam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "convnext_tiny"
CHECKPOINT_PATH = "models/baseline_5epochs.pth"
DATASET_REPO_ID = CUSTOM_TEST_REPO_ID
DATASET_SPLIT = CUSTOM_TEST_SPLIT
OUTPUT_DIR = "xai_outputs/grad_cam"
NUM_IMAGES = 8
SAMPLE_MODE = "balanced"  # "balanced" or "first"
TARGET_CLASS_MODE = "predicted"  # "predicted" or "true"

CHECKPOINTS = {
    "baseline": "models/baseline_5epochs.pth",
    "convnext_tiny": "models/convnext_tiny_3epochs.pth",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_class_names():
    final_classes_path = PROJECT_ROOT / "configs" / "final_classes.json"
    with final_classes_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return len(class_names), class_names


def create_model(num_classes: int):
    if MODEL_NAME == "baseline":
        from src.models.resnet import resnet18

        return resnet18(num_classes=num_classes)

    if MODEL_NAME == "convnext_tiny":
        from src.models.convnext_tiny import convnext_tiny

        return convnext_tiny(num_classes=num_classes)

    raise ValueError(
        f"Unsupported Grad-CAM model: {MODEL_NAME}. "
        "Supported models: baseline, convnext_tiny."
    )


def load_model(num_classes: int):
    checkpoint_path = CHECKPOINTS.get(MODEL_NAME, CHECKPOINT_PATH)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(PROJECT_ROOT / checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, checkpoint_path


def select_examples(dataset, num_classes: int):
    if SAMPLE_MODE == "first":
        return dataset.select(range(min(NUM_IMAGES, len(dataset))))

    examples = []
    class_counts = {class_id: 0 for class_id in range(num_classes)}
    max_per_class = max(1, (NUM_IMAGES + num_classes - 1) // num_classes)

    for item in dataset:
        label = int(item["label"])
        if class_counts[label] >= max_per_class:
            continue

        examples.append(item)
        class_counts[label] += 1

        if len(examples) >= NUM_IMAGES:
            break

    return examples


def main():
    num_classes, class_names = load_class_names()
    model, checkpoint_path = load_model(num_classes)
    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {DATASET_REPO_ID} [{DATASET_SPLIT}]")

    dataset = load_dataset(DATASET_REPO_ID, split=DATASET_SPLIT)
    transform = model.transforms

    try:
        for idx, item in enumerate(select_examples(dataset, num_classes)):
            image = prepare_rgb_image(item["image"])
            label = int(item["label"])
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            target_class = label if TARGET_CLASS_MODE == "true" else None
            heatmap, predicted_label, cam_label, confidence = grad_cam(
                input_tensor,
                target_class=target_class,
            )
            output_path = Path(
                f"{PROJECT_ROOT / OUTPUT_DIR}/"
                f"{MODEL_NAME}_{DATASET_SPLIT}_{idx:03d}_"
                f"cam-{TARGET_CLASS_MODE}_"
                f"true-{class_names[label]}_"
                f"pred-{class_names[predicted_label]}_"
                f"target-{class_names[cam_label]}_"
                f"conf-{confidence:.2f}.jpg"
            )

            save_grad_cam(
                image=image,
                heatmap=heatmap,
                output_path=output_path,
            )
            print(output_path)
    finally:
        grad_cam.remove_hooks()


if __name__ == "__main__":
    main()
