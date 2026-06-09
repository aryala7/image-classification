"""Household Garbage dataset: download coordinates and sample collection."""
from pathlib import Path
import csv

from .common import add_sample, is_image_file, load_json

NAME = "household_garbage"
KAGGLE_ID = "alistairking/recyclable-and-household-waste-classification"
KAGGLE_SUBPATH = "images/images"


def _is_downloaded(raw_data_dir: Path) -> bool:
    """True if the raw dataset folder exists and is not empty."""
    dataset_path = raw_data_dir / NAME
    return dataset_path.is_dir() and any(dataset_path.iterdir())


def _is_harmonized(metadata_path: Path) -> bool:
    """True if metadata.csv already contains rows for this dataset."""
    if not metadata_path.exists():
        return False
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        return any(row["source_dataset"] == NAME for row in csv.DictReader(file))


def collect(
        dataset_path: Path,
        final_classes: list[str],
        stats: dict,
        project_root: Path
) -> list[dict]:
    label_mapping = load_json(project_root / "configs" / f"label_mapping_{NAME}.json")
    samples = []
    source_dataset = NAME
    for label_dir in dataset_path.iterdir():
        if not label_dir.is_dir():
            continue
        original_label = label_dir.name
        for context in ["default", "real_world"]:
            context_dir = label_dir / context
            if not context_dir.exists():
                continue
            for image_path in context_dir.rglob("*"):
                if image_path.is_file() and is_image_file(image_path):
                    add_sample(
                        samples=samples,
                        image_path=image_path,
                        source_dataset=source_dataset,
                        original_label=original_label,
                        image_context=context,
                        label_mapping=label_mapping,
                        final_classes=final_classes,
                        stats=stats,
                        project_root=project_root
                    )
    return samples
