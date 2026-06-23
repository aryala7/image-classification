import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Image, ClassLabel


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_metadata(df: pd.DataFrame, final_classes: list[str]) -> None:
    """
    Validate metadata before creating the Hugging Face dataset.
    """
    required_columns = {
        "image_path",
        "label",
        "label_id",
        "source_dataset",
        "original_label",
        "image_context",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in metadata.csv: {missing_columns}")
    unknown_labels = sorted(set(df["label"]) - set(final_classes))
    if unknown_labels:
        raise ValueError(f"Unknown labels found in metadata.csv: {unknown_labels}")
    missing_files = [
        image_path
        for image_path in df["image_path"].tolist()
        if not Path(image_path).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} image files do not exist. "
            f"First missing files: {missing_files[:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Upload harmonized dataset to HF Hub")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to harmonized dataset folder containing metadata.csv")
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="Hugging Face dataset repository name")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    final_classes_path = project_root / "configs" / "final_classes.json"
    final_classes = load_json(final_classes_path)

    metadata_path = args.dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)

    validate_metadata(df, final_classes)

    # Create absolute/local image paths for Hugging Face Image column
    df["image"] = df["image_path"]

    # Make sure label_id is integer
    df["label_id"] = df["label_id"].astype(int)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    # Cast image column so Hugging Face loads images as PIL images
    dataset = dataset.cast_column("image", Image())
    # Cast label_id column as ClassLabel for cleaner HF dataset features
    dataset = dataset.cast_column(
        "label_id",
        ClassLabel(names=final_classes)
    )
    dataset_dict = DatasetDict({
        "inference": dataset
    })

    print(dataset_dict)
    print(dataset_dict["inference"].features)
    print(dataset_dict["inference"][0])

    dataset_dict.push_to_hub(args.hf_dataset_name)

    print(f"Uploaded dataset to: {args.hf_dataset_name}")


if __name__ == "__main__":
    main()