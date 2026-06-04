## Goal

The goal of this pipeline is to combine the following datasets:

- TrashNet
- Household Garbage Dataset
- The Garbage Dataset (GD)

The final label system is based on the TrashNet classes:

```text
cardboard
glass
metal
paper
plastic
trash
```

More specific labels from the other datasets are mapped into these broader classes. For example:

```text
plastic_cup_lids        -> plastic
plastic_food_containers -> plastic
glass_bottles           -> glass
metal_cans              -> metal
```

Labels that do not clearly match the final label space should be marked as `exclude` in the label mapping configuration.

---

## Project Structure

```text
project/
├── configs/
│   ├── final_classes.json
│   └── label_mapping.json
│
├── data/
│   ├── raw/
│   │   ├── trashnet/
│   │   ├── household_garbage/
│   │   └── garbage_dataset/
│   │
│   └── processed/
│       └── combined_waste_dataset/
│           └── metadata.csv
│
├── reports/
│   └── harmonization_report.md
│
└── src/
    └── data/
        └── harmonize_datasets.py
```

The `data/raw/` directory contains the original datasets and should not be modified manually.

The `data/processed/` directory contains generated files created by the preprocessing pipeline.

---

## Raw Dataset Structure

### TrashNet

Expected structure:

```text
data/raw/trashnet/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Each folder name is treated as the original label.

---

### Household Garbage Dataset

Expected structure:

```text
data/raw/household_garbage/
├── plastic_cup_lids/
│   ├── default/
│   └── real_world/
├── plastic_food_containers/
│   ├── default/
│   └── real_world/
└── ...
```

Each label folder contains two image contexts:

- `default`: studio-like or controlled images
- `real_world`: images from realistic environments

Both contexts are included in the combined dataset. The context is stored in the metadata as `image_context`.

---

### The Garbage Dataset (GD)

The GD dataset may contain multiple variants, for example:

```text
data/raw/garbage_dataset/
├── original/
├── standardized_256/
└── standardized_384/
```

For this project, only the original images should be used:

```text
data/raw/garbage_dataset/original/
```

The standardized folders should not be combined with the original folder because they may contain resized versions of the same images. Using all variants together could create duplicate data and cause train/test leakage.

---

## Configuration Files

### `configs/final_classes.json`

This file defines the final class names used for training.

Example:

```json
[
  "cardboard",
  "glass",
  "metal",
  "paper",
  "plastic",
  "trash"
]
```

---

### `configs/label_mapping.json`

This file maps source dataset labels to the final labels.

Example:

```json
{
  "trashnet": {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash"
  },
  "household_garbage": {
    "plastic_cup_lids": "plastic",
    "plastic_food_containers": "plastic",
    "glass_bottles": "glass",
    "metal_cans": "metal",
    "food_waste": "exclude"
  },
  "garbage_dataset": {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash"
  }
}
```

If a label should not be used, map it to:

```json
"exclude"
```

This prevents unclear or incompatible labels from adding noise to the dataset.

---

## Harmonisation Pipeline

The main script is:

```text
src/data/harmonize_datasets.py
```

It performs the following steps:

1. Loads the final class list from `configs/final_classes.json`
2. Loads the label mapping from `configs/label_mapping.json`
3. Reads all images from the raw datasets
4. Maps each original label to the final TrashNet-based label system
5. Excludes labels marked as `exclude`
6. Stores metadata for each valid image
7. Generates `metadata.csv`
8. Generates a harmonisation report

Run the script from the project root:

```bash
python src/data/harmonize_datasets.py
```

---

## Generated Metadata

The script creates:

```text
data/processed/combined_waste_dataset/metadata.csv
```

Each row contains:

```text
image_path,label,label_id,source_dataset,original_label,image_context
```

Example:

```text
data/raw/trashnet/plastic/img1.jpg,plastic,4,trashnet,plastic,default
data/raw/household_garbage/plastic_cup_lids/default/img2.jpg,plastic,4,household_garbage,plastic_cup_lids,default
data/raw/household_garbage/plastic_cup_lids/real_world/img3.jpg,plastic,4,household_garbage,plastic_cup_lids,real_world
data/raw/garbage_dataset/original/plastic/img4.jpg,plastic,4,garbage_dataset,plastic,unknown
```

The metadata keeps important information about each image:

- `image_path`: relative path to the image
- `label`: final harmonised label
- `label_id`: numeric label ID
- `source_dataset`: original dataset name
- `original_label`: original folder label
- `image_context`: image context such as `default`, `real_world`, or `unknown`

---

## Harmonisation Report

The script also creates:

```text
reports/harmonization_report.md
```

The report contains:

- number of valid images
- number of excluded images
- final label distribution
- source dataset distribution
- unknown labels
- invalid mapped labels

This report should be checked after every dataset update.

---

## Uploading to Hugging Face

After harmonisation, the `metadata.csv` can be converted into a Hugging Face `DatasetDict` with three splits:

```text
train
validation
test
```

The recommended workflow is:

```text
raw datasets
    ↓
harmonize_datasets.py
    ↓
metadata.csv
    ↓
create_hf_dataset.py
    ↓
push_to_hub()
```

The Hugging Face dataset should preserve the following columns:

```text
image
label
label_id
source_dataset
original_label
image_context
```

The dataset should initially be uploaded as private until all source dataset licenses are checked.

---

## Adding New Data

New data should not be added manually to the processed dataset or directly to Hugging Face.

Instead, new images should be added to the raw data folder and the pipeline should be executed again.

---

### Option 1: Add More Images to an Existing Label

If the new images already match one of the final classes, place them under a custom dataset folder:

```text
data/raw/custom_images/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Example:

```text
data/raw/custom_images/plastic/my_plastic_bottle_01.jpg
data/raw/custom_images/plastic/my_plastic_cup_02.jpg
data/raw/custom_images/paper/my_paper_01.jpg
```

Then add `custom_images` to `configs/label_mapping.json`:

```json
{
  "custom_images": {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash"
  }
}
```

The harmonisation script then needs a collector for the custom images source.

---

### Option 2: Add a New Source Dataset

If a completely new dataset is added, use this structure:

```text
data/raw/new_dataset_name/
├── original_label_1/
├── original_label_2/
└── original_label_3/
```

Then add a new section to `configs/label_mapping.json`:

```json
{
  "new_dataset_name": {
    "original_label_1": "plastic",
    "original_label_2": "paper",
    "original_label_3": "exclude"
  }
}
```

If the folder structure is different from the existing datasets, a new collector function should be added to `harmonize_datasets.py`.

---

### Recommended Update Workflow

Whenever new data is added, follow this process:

```text
1. Add new images under data/raw/
2. Update configs/label_mapping.json if needed
3. Run harmonize_datasets.py
4. Check reports/harmonization_report.md
5. Recreate train/validation/test splits
6. Recreate the Hugging Face DatasetDict
7. Push the updated dataset to the same Hugging Face repository
8. Update the dataset card/version notes
```
