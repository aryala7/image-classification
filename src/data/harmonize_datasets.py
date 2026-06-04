from pathlib import Path
import json
import csv

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
        stats: dict
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


def collect_trashnet(
        dataset_path: Path,
        label_mapping: dict,
        final_classes: list[str],
        stats: dict
) -> list[dict]:
    samples = []
    source_dataset = "trashnet"
    for label_dir in dataset_path.iterdir():
        if not label_dir.is_dir():
            continue
        original_label = label_dir.name
        for image_path in label_dir.rglob("*"):
            if image_path.is_file() and is_image_file(image_path):
                add_sample(
                    samples=samples,
                    image_path=image_path,
                    source_dataset=source_dataset,
                    original_label=original_label,
                    image_context="default",
                    label_mapping=label_mapping,
                    final_classes=final_classes,
                    stats=stats

                )

    return samples


def collect_household_garbage(
        dataset_path: Path,
        label_mapping: dict,
        final_classes: list[str],
        stats: dict
) -> list[dict]:
    samples = []
    source_dataset = "household_garbage"
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
                        stats=stats
                    )
    return samples


def collect_garbage_dataset(
        dataset_path: Path,
        label_mapping: dict,
        final_classes: list[str],
        stats: dict
) -> list[dict]:
    samples = []
    source_dataset = "garbage_dataset"
    for label_dir in dataset_path.iterdir():
        if not label_dir.is_dir():
            continue
        original_label = label_dir.name
        for image_path in label_dir.rglob("*"):
            if image_path.is_file() and is_image_file(image_path):
                add_sample(
                    samples=samples,
                    image_path=image_path,
                    source_dataset=source_dataset,
                    original_label=original_label,
                    image_context="unknown",
                    label_mapping=label_mapping,
                    final_classes=final_classes,
                    stats=stats
                )
    return samples


def write_metadata_csv(samples: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "label_id",
        "source_dataset",
        "original_label",
        "image_context"
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def write_report(samples: list[dict], stats: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = {}

    source_counts = {}

    for sample in samples:
        label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1

        source_counts[sample["source_dataset"]] = source_counts.get(sample["source_dataset"], 0) + 1

    with output_path.open("w", encoding="utf-8") as file:

        file.write("# Harmonization Report\n\n")
        file.write(f"Valid images: {stats['valid_images']}\n\n")
        file.write(f"Excluded images: {stats['excluded']}\n\n")
        file.write("## Label Distribution\n\n")
        file.write("| Label | Count |\n")
        file.write("|---|---:|\n")
        for label, count in sorted(label_counts.items()):
            file.write(f"| {label} | {count} |\n")
        file.write("\n## Source Dataset Distribution\n\n")
        file.write("| Source Dataset | Count |\n")
        file.write("|---|---:|\n")
        for source, count in sorted(source_counts.items()):
            file.write(f"| {source} | {count} |\n")
        file.write("\n## Unknown Labels\n\n")
        if stats["unknown_labels"]:
            for source_dataset, label in sorted(stats["unknown_labels"]):
                file.write(f"- {source_dataset}: {label}\n")
        else:
            file.write("No unknown labels.\n")
        file.write("\n## Invalid Mapped Labels\n\n")
        if stats["invalid_mapped_labels"]:
            for label in sorted(stats["invalid_mapped_labels"]):
                file.write(f"- {label}\n")
        else:
            file.write("No invalid mapped labels.\n")


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed" / "combined_waste_dataset"
    label_mapping = load_json(project_root / "configs" / "label_mapping.json")
    final_classes = load_json(project_root / "configs" / "final_classes.json")

    stats = {
        "valid_images": 0,
        "excluded": 0,
        "broken_images": 0,
        "unknown_labels": set(),
        "invalid_mapped_labels": set()
    }
    all_samples = []
    all_samples.extend(
        collect_trashnet(
            dataset_path=raw_data_dir / "trashnet",
            label_mapping=label_mapping,
            final_classes=final_classes,
            stats=stats
        )
    )
    all_samples.extend(
        collect_household_garbage(
            dataset_path=raw_data_dir / "household_garbage",
            label_mapping=label_mapping,
            final_classes=final_classes,
            stats=stats
        )
    )
    all_samples.extend(
        collect_garbage_dataset(
            dataset_path=raw_data_dir / "garbage_dataset",
            label_mapping=label_mapping,
            final_classes=final_classes,
            stats=stats
        )
    )
    write_metadata_csv(
        samples=all_samples,
        output_path=processed_data_dir / "metadata.csv"
    )
    write_report(
        samples=all_samples,
        stats=stats,
        output_path=project_root / "reports" / "harmonization_report.md"
    )
    print(f"Created metadata for {len(all_samples)} valid images.")
    print(f"Metadata saved to: {processed_data_dir / 'metadata.csv'}")
    print(f"Report saved to: {project_root / 'reports' / 'harmonization_report.md'}")

if __name__ == "__main__":
    main()