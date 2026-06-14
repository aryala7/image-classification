from pathlib import Path
import csv
import hashlib
from collections import defaultdict

from dataset_sources import ALL
from dataset_sources.common import load_json

MAX_SAMPLES_PER_LABEL = 5000
LABELS_TO_KEEP_ALL = {"cardboard"}
PREFERRED_CONTEXTS = {"real_world"}
SAMPLING_SEED = "waste-harmonization-v1"


def read_metadata_by_source(metadata_path: Path) -> dict:
    """Group existing metadata.csv rows by their source_dataset."""
    rows_by_source = {}
    if not metadata_path.exists():
        return rows_by_source
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows_by_source.setdefault(row["source_dataset"], []).append(row)
    return rows_by_source


def stable_sample_key(sample: dict) -> str:
    """Return a deterministic pseudo-random key for reproducible sampling."""
    identity = "|".join(
        [
            SAMPLING_SEED,
            sample["label"],
            sample["source_dataset"],
            sample["original_label"],
            sample["image_context"],
            sample["image_path"],
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def allocate_stratified_counts(strata: dict[tuple, list[dict]], target_count: int) -> dict[tuple, int]:
    """Allocate a target count proportionally across strata using largest remainders."""
    total = sum(len(samples) for samples in strata.values())
    if target_count >= total:
        return {key: len(samples) for key, samples in strata.items()}

    quotas = {}
    remaining = target_count
    remainders = []
    for key, samples in sorted(strata.items()):
        exact_quota = target_count * len(samples) / total
        count = int(exact_quota)
        quotas[key] = count
        remaining -= count
        remainders.append((exact_quota - count, len(samples), key))

    for _, _, key in sorted(remainders, reverse=True)[:remaining]:
        quotas[key] += 1

    return quotas


def stratified_downsample(samples: list[dict], target_count: int) -> list[dict]:
    """Downsample while preserving source, context, and original-label diversity."""
    if len(samples) <= target_count:
        return samples

    strata = defaultdict(list)
    for sample in samples:
        key = (
            sample["source_dataset"],
            sample["image_context"],
            sample["original_label"],
        )
        strata[key].append(sample)

    quotas = allocate_stratified_counts(strata, target_count)
    selected = []
    for key, stratum_samples in strata.items():
        selected.extend(
            sorted(stratum_samples, key=stable_sample_key)[:quotas[key]]
        )

    return selected


def downsample_label_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    """Cap each non-exempt label and preferentially preserve real-world images."""
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)

    downsampled = []
    sampling_stats = {}
    for label, label_samples in sorted(by_label.items()):
        original_count = len(label_samples)
        if label in LABELS_TO_KEEP_ALL or original_count <= MAX_SAMPLES_PER_LABEL:
            kept = label_samples
        else:
            preferred = [
                sample
                for sample in label_samples
                if sample["image_context"] in PREFERRED_CONTEXTS
            ]
            other = [
                sample
                for sample in label_samples
                if sample["image_context"] not in PREFERRED_CONTEXTS
            ]
            if len(preferred) >= MAX_SAMPLES_PER_LABEL:
                kept = stratified_downsample(preferred, MAX_SAMPLES_PER_LABEL)
            else:
                kept = preferred + stratified_downsample(
                    other,
                    MAX_SAMPLES_PER_LABEL - len(preferred),
                )

        kept = sorted(kept, key=lambda sample: sample["image_path"])
        downsampled.extend(kept)
        sampling_stats[label] = {
            "original_count": original_count,
            "kept_count": len(kept),
            "removed_count": original_count - len(kept),
        }

    return downsampled, sampling_stats


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
        file.write(f"Collected valid images: {stats['valid_images']}\n\n")
        file.write(f"Harmonized images after downsampling: {len(samples)}\n\n")
        file.write(f"Excluded images: {stats['excluded']}\n\n")
        file.write("## Label Distribution\n\n")
        file.write("| Label | Count |\n")
        file.write("|---|---:|\n")
        for label, count in sorted(label_counts.items()):
            file.write(f"| {label} | {count} |\n")
        file.write("\n## Downsampling Summary\n\n")
        file.write("| Label | Original | Kept | Removed |\n")
        file.write("|---|---:|---:|---:|\n")
        for label, sampling in sorted(stats["sampling"].items()):
            file.write(
                f"| {label} | {sampling['original_count']} | "
                f"{sampling['kept_count']} | {sampling['removed_count']} |\n"
            )
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
        "invalid_mapped_labels": set(),
        "sampling": {}
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
        dataset_path = raw_data_dir / source.NAME
        if not source._is_downloaded(raw_data_dir):
            print(f"Skipping {source.NAME}: raw dataset not found at {dataset_path}")
            continue
        all_samples.extend(
            source.collect(
                dataset_path=dataset_path,
                final_classes=final_classes,
                stats=stats,
                project_root=project_root
            )
        )
    if not all_samples:
        raise RuntimeError(
            f"No images were collected from {raw_data_dir}. "
            "Download or copy the raw datasets before harmonizing."
        )

    harmonized_samples, stats["sampling"] = downsample_label_samples(all_samples)

    write_metadata_csv(
        samples=harmonized_samples,
        output_path=metadata_path
    )
    write_report(
        samples=harmonized_samples,
        stats=stats,
        output_path=project_root / "reports" / "harmonization_report.md"
    )
    print(f"Collected {len(all_samples)} valid images.")
    print(f"Created metadata for {len(harmonized_samples)} harmonized images.")
    print(f"Metadata saved to: {processed_data_dir / 'metadata.csv'}")
    print(f"Report saved to: {project_root / 'reports' / 'harmonization_report.md'}")

if __name__ == "__main__":
    main()
