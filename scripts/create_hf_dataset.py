from pathlib import Path
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, ClassLabel, Value
from sklearn.model_selection import train_test_split
from huggingface_hub import login

REPO_ID = "TrashBusters/combined"
PUSH_TO_HUB = False
PRIVATE_DATASET = True
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


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


def create_split(df: pd.DataFrame):
    train_df,temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )
    validation_size = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_df,test_df = train_test_split(
        temp_df,
        train_size=validation_size,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def print_split_distribution(df: pd.DataFrame, split_name:str):
    print(f"\n{split_name} label distribution:")
    print(df["label"].value_counts().sort_index())
def dataframe_to_hf(
        df: pd.DataFrame,
        final_classes: list[str]
):
    df = df.copy()

    # Keep the readable class name for debugging and analysis.
    df["label_name"] = df["label"]

    # ClassLabel expects the label column to contain integer class IDs.
    df["label"] = df["label"].apply(lambda label: final_classes.index(label))

    # Keep label_id consistent with the ClassLabel integer value.
    df["label_id"] = df["label"]

    # Hugging Face expects the image column to be called image.
    df = df.rename(columns={"image_path": "image"})

    features = Features({
        "image": Image(),
        "label": ClassLabel(names=final_classes),
        "label_id": Value("int64"),
        "label_name": Value("string"),
        "source_dataset": Value("string"),
        "original_label": Value("string"),
        "image_context": Value("string")
    })

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.cast(features)
    return dataset

def main():
    project_root = Path(__file__).resolve().parents[1]
    matadata_path = (project_root
                     / "data"
                     / "processed"
                     / "combined_waste_dataset"
                     / "metadata.csv"
                     )
    final_classes_path = project_root / "configs" / "final_classes.json"
    local_output_path = (
            project_root / "data" / "processed" / "hf_combined_waste_dataset"
    )
    final_classes = load_json(final_classes_path)
    df = pd.read_csv(matadata_path)
    validate_metadata(df, final_classes)
    train_df,validation_df,test_df = create_split(df)
    print_split_distribution(train_df, "Train")
    print_split_distribution(validation_df, "Validation")
    print_split_distribution(test_df, "Test")
    dataset_dict = DatasetDict({
        "train": dataframe_to_hf(train_df,final_classes),
        "validation": dataframe_to_hf(validation_df,final_classes),
        "test": dataframe_to_hf(test_df,final_classes)
    })

    print(f"\nCreated Hugging Face DatasetDict:")
    print(dataset_dict)
    local_output_path.mkdir(parents=True,exist_ok=True)
    dataset_dict.save_to_disk(str(local_output_path))

    if PUSH_TO_HUB:
        dataset_dict.push_to_hub(
            REPO_ID,
            private=PRIVATE_DATASET
        )
        print(f"\nUploaded dataset to Hugging Face:")
        print(REPO_ID)


if __name__ == "__main__":
    if PUSH_TO_HUB:
        login()
    main()
