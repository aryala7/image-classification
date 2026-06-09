"""Smoke test data loader — TrashNet only, reads from merged CSV."""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

CSV_PATH = Path.cwd() / "data" / "processed" / "combined_waste_dataset" / "metadata.csv"


class CSVDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label_id"]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_loaders(transform, batch_size: int) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(CSV_PATH)
    df = df[df["source_dataset"] == "trashnet"]

    dataset = CSVDataset(df, transform=transform)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_set, val_set, _ = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader