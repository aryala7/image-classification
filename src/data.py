"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def load_image_folder_dataset(
    data_dir: str,
    transform: transforms.Compose,
) -> datasets.ImageFolder:
    """Load an ImageFolder dataset from a directory.

    The directory must follow the ImageFolder structure:
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, but got: {data_path}")

    return datasets.ImageFolder(
        root=data_path,
        transform=transform,
    )


def split_dataset_indices(
    labels: List[int],
    train_size: float = 0.75,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Create stratified train and validation indices.

    Stratification keeps the class distribution similar in both splits.
    The function only needs labels, so it works for ImageFolder and Hugging Face datasets.
    """
    train_indices, val_indices = train_test_split(
        range(len(labels)),
        train_size=train_size,
        stratify=labels,
        random_state=seed,
    )

    return list(train_indices), list(val_indices)


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 1,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Create a PyTorch DataLoader for a given dataset or subset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def create_imagefolder_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 1,
    seed: int = 42,
    train_size: float = 0.75,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation dataloaders from an ImageFolder dataset."""
    dataset = load_image_folder_dataset(
        data_dir=data_dir,
        transform=transform,
    )

    train_indices, val_indices = split_dataset_indices(
        labels=dataset.targets,
        train_size=train_size,
        seed=seed,
    )

    train_loader = create_data_loader(
        dataset=Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_data_loader(
        dataset=Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, dataset.classes


def create_hf_collate_fn(transform: transforms.Compose) -> Callable:
    """Create a collate function for Hugging Face image datasets.

    Hugging Face datasets usually return dictionaries containing PIL images and labels.
    The collate function converts images and labels into tensors that the model can use.
    """

    def collate_fn(batch):
        images = [
            transform(item["image"].convert("RGB"))
            for item in batch
        ]

        labels = [
            item["label"]
            for item in batch
        ]

        return {
            "images": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def create_hf_dataloaders(
    dataset: Dataset,
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 1,
    seed: int = 42,
    train_size: float = 0.75,
    label_column: str = "label",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a Hugging Face dataset.

    The dataset must contain an image column named "image" and a label column.
    By default, the label column is expected to be named "label".
    """
    labels = list(dataset[label_column])

    train_indices, val_indices = split_dataset_indices(
        labels=labels,
        train_size=train_size,
        seed=seed,
    )

    collate_fn = create_hf_collate_fn(transform)

    train_loader = create_data_loader(
        dataset=Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = create_data_loader(
        dataset=Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
