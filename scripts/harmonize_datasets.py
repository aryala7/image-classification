from pathlib import Path
import csv

from dataset_sources import ALL
from dataset_sources.common import load_json


def read_metadata_by_source(metadata_path: Path) -> dict:
    """Group existing metadata.csv rows by their source_dataset."""
    rows_by_source = {}
    if not metadata_path.exists():
        return rows_by_source
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows_by_source.setdefault(row["source_dataset"], []).append(row)
    return rows_by_source


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
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed" / "combined_waste_dataset"
    final_classes = load_json(project_root / "configs" / "final_classes.json")

    stats = {
        "valid_images": 0,
        "excluded": 0,
        "broken_images": 0,
        "unknown_labels": set(),
        "invalid_mapped_labels": set()
    }
    metadata_path = processed_data_dir / "metadata.csv"
    existing_rows_by_source = read_metadata_by_source(metadata_path)

    all_samples = []
    for source in ALL:
        if source._is_harmonized(metadata_path):
            reused = existing_rows_by_source.get(source.NAME, [])
            stats["valid_images"] += len(reused)
            all_samples.extend(reused)
            print(f"Skipping {source.NAME}: already harmonized ({len(reused)} samples reused)")
            continue
        all_samples.extend(
            source.collect(
                dataset_path=raw_data_dir / source.NAME,
                final_classes=final_classes,
                stats=stats,
                project_root=project_root
            )
        )
    write_metadata_csv(
        samples=all_samples,
        output_path=metadata_path
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