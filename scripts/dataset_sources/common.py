"""Shared helpers used by the per-dataset collector modules."""
from pathlib import Path
import json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_label_id(label: str, final_classes: list[str]) -> int:
    return final_classes.index(label)


def add_sample(
        samples: list[dict],
        image_path: Path,
        source_dataset: str,
        original_label: str,
        image_context: str,
        label_mapping: dict,
        final_classes: list[str],
        stats: dict,
        project_root: Path
):
    mapped_label = label_mapping[source_dataset].get(original_label)
    if mapped_label is None:
        stats["unknown_labels"].add((source_dataset, original_label))
        return

    if mapped_label == "exclude":
        stats["excluded"] += 1
        return
    if mapped_label not in final_classes:
        stats["invalid_mapped_labels"].add(mapped_label)
        return

    samples.append({
        "image_path": str(image_path),
        "label": mapped_label,
        "label_id": get_label_id(mapped_label, final_classes),
        "source_dataset": source_dataset,
        "original_label": original_label,
        "image_context": image_context
    })
    stats['valid_images'] += 1
