"""Data loading utilities for the Hugging Face waste dataset."""

from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

HF_REPO_ID = "TrashBusters/combined"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"


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


def create_training_dataloaders(
    transform: transforms.Compose,
    batch_size: int,
    data_config: dict,
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
