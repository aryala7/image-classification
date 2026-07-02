"""Data loading utilities for the Hugging Face waste dataset."""

from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

HF_REPO_ID = "TrashBusters/combined"
CUSTOM_TEST_REPO_ID = "TrashBusters/wild-trash-inference-dataset"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
CUSTOM_TEST_SPLIT = "inference"


def prepare_rgb_image(image: Image.Image) -> Image.Image:
    """Convert PIL images to RGB without palette-transparency warnings."""
    if image.mode == "RGB":
        return image

    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return image.convert("RGB")


def create_hf_collate_fn(transform: transforms.Compose) -> Callable:
    """Create a collate function that returns batches as ``(images, labels)``."""

    def collate_fn(batch):
        images = [
            transform(prepare_rgb_image(item["image"]))
            for item in batch
        ]
        labels = [
            item["label"]
            for item in batch
        ]

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    return collate_fn


def create_data_loader(
    dataset,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a PyTorch DataLoader for one Hugging Face split."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=create_hf_collate_fn(transform),
    )


def _resolve_exclude_ids(dataset_split, exclude_classes) -> set:
    """Map class names to their integer label ids for a given split.

    Prefers the split's ClassLabel feature (str2int); falls back to the
    canonical ordering in ``engine.CLASS_NAMES`` (shared across the combined
    and custom datasets) when the feature is not a ClassLabel.
    """
    feature = dataset_split.features.get("label")
    exclude_ids: set = set()
    for name in exclude_classes:
        resolved = None
        if hasattr(feature, "str2int"):
            try:
                resolved = feature.str2int(name)
            except (KeyError, ValueError):
                resolved = None
        if resolved is None:
            from src.engine import CLASS_NAMES

            if name in CLASS_NAMES:
                resolved = CLASS_NAMES.index(name)
        if resolved is not None:
            exclude_ids.add(resolved)
    return exclude_ids


def _filter_out_classes(dataset, reference_split, exclude_classes):
    """Return ``dataset`` with all rows whose label is in ``exclude_classes``
    removed. Works on a single split or a DatasetDict. No-op when nothing
    resolves. ``reference_split`` supplies the ClassLabel mapping."""
    if not exclude_classes:
        return dataset

    exclude_ids = _resolve_exclude_ids(reference_split, exclude_classes)
    if not exclude_ids:
        print(
            f"[data] Warning: could not resolve classes {exclude_classes} to label "
            f"ids; no rows excluded."
        )
        return dataset

    filtered = dataset.filter(lambda example: example["label"] not in exclude_ids)
    print(
        f"[data] Excluded classes {list(exclude_classes)} "
        f"(label ids {sorted(exclude_ids)})."
    )
    return filtered


def create_training_dataloaders(
    transform: transforms.Compose,
    batch_size: int,
    data_config: dict,
    exclude_classes: list[str] | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    repo_id = data_config.get("repo_id", HF_REPO_ID)
    num_workers = data_config.get("num_workers", 0)

    print(f"Loading Hugging Face Hub dataset {repo_id}...")
    dataset = load_dataset(repo_id)
    expected_splits = {TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT}
    missing_splits = expected_splits - set(dataset.keys())
    if missing_splits:
        available_splits = ", ".join(dataset.keys())
        raise KeyError(
            f"Missing Hugging Face split(s): {sorted(missing_splits)}. "
            f"Available splits: [{available_splits}]"
        )

    dataset = _filter_out_classes(dataset, dataset[TRAIN_SPLIT], exclude_classes)

    print(f"Using Hugging Face Hub dataset {repo_id}")
    train_loader = create_data_loader(
        dataset=dataset[TRAIN_SPLIT],
        transform=transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = create_data_loader(
        dataset=dataset[VALIDATION_SPLIT],
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = create_data_loader(
        dataset=dataset[TEST_SPLIT],
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader


def create_custom_test_dataloader(
    transform: transforms.Compose,
    batch_size: int,
    data_config: dict,
    exclude_classes: list[str] | None = None,
) -> DataLoader:
    """Create a DataLoader for the custom wild-trash inference dataset."""
    num_workers = data_config.get("num_workers", 0)

    print(f"Loading Hugging Face Hub dataset {CUSTOM_TEST_REPO_ID}...")
    dataset = load_dataset(CUSTOM_TEST_REPO_ID, split=CUSTOM_TEST_SPLIT)
    dataset = _filter_out_classes(dataset, dataset, exclude_classes)
    print(f"Using Hugging Face Hub dataset {CUSTOM_TEST_REPO_ID}")

    return create_data_loader(
        dataset=dataset,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
