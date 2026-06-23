import json
from pathlib import Path
import csv

project_root = Path(__file__).resolve().parents[1]
raw_data_dir = project_root / "data" / "raw" / "custom" / "Trash"
DATASET_DIR = raw_data_dir
OUTPUT_FILE = DATASET_DIR / "metadata.csv"
final_classes_path = project_root / "configs" / "final_classes.json"

with final_classes_path.open("r", encoding="utf-8") as file:
    final_classes = json.load(file)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

rows = []

for label_id, label_name in enumerate(final_classes):
    label_dir = DATASET_DIR / label_name

    if not label_dir.exists():
        print(f"Skipping missing folder: {label_dir}")
        continue

    for image_path in sorted(label_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            rows.append({
                "image_path": str(image_path),
                "label": label_name,
                "label_id": label_id,
                "source_dataset": "wild_trash",
                "original_label": label_name,
                "image_context": "real_world",
            })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "image_path",
        "label",
        "label_id",
        "source_dataset",
        "original_label",
        "image_context",
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {OUTPUT_FILE} with {len(rows)} rows.")