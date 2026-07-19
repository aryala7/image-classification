# TrashBusters — Waste Image Classification

TrashBusters classifies waste images into six categories (cardboard, glass, metal, paper, plastic, trash). It combines four public waste-classification datasets into a single harmonised label space, trains and compares several CNN/transformer architectures on the result, and uses explainability methods (Grad-CAM, Integrated Gradients, Occlusion) to inspect what the models are actually keying on. Model robustness is additionally checked against a small custom, real-world image set to measure performance under domain shift.

## Repository Structure

```text
.
├── configs/     # Project/training config (config.yaml), label mappings, final class list
├── data/        # Raw and processed/harmonised datasets (populated by scripts/, not versioned)
├── docs/        # Dataset overview and harmonisation pipeline documentation
├── models/      # Fine-tuned model checkpoints (.pth)
├── notebooks/   # Evaluation notebooks: metrics comparison, class analysis, dataset charts
├── reports/     # Generated harmonisation report (label distribution, exclusions, etc.)
├── scripts/     # Dataset download/harmonisation, HF export, and XAI scripts
├── src/         # Core package: data loading, models, training engine, fine-tuning, XAI utilities
└── main.py      # Interactive CLI (train / test / test_custom) entry point
```

## Conda Environment

Create the environment:

```bash
conda env create -f environment.yml
```

Update the environment, **depending on cuda support**:

```bash
conda env update -f environment-<cuda/cpu>.yml
```

Activate it:

```bash
conda activate base-project
```

Update it after changing dependencies:

```bash
conda env update -f environment.yml --prune
```

## Datasets Used

This project combines and harmonises the following public waste-classification datasets into a shared six-class label space (cardboard, glass, metal, paper, plastic, trash). See [docs/dataset_harmonization.md](docs/dataset_harmonization.md) for the full label-mapping pipeline and [reports/harmonization_report.md](reports/harmonization_report.md) for the resulting class/source distribution.

The harmonised result is published on Hugging Face and is what training/evaluation actually loads (via `data.repo_id` in [configs/config.yaml](configs/config.yaml)):

- **[TrashBusters/combined](https://huggingface.co/datasets/TrashBusters/combined)** — the harmonised train/validation/test dataset used for training and in-distribution evaluation
- **[TrashBusters/wild-trash-inference-dataset](https://huggingface.co/datasets/TrashBusters/wild-trash-inference-dataset)** — the custom, real-world image set used for the domain-shift evaluation in [Results](#results)

> The raw-dataset and harmonisation pipeline below (`scripts/download_datasets.py`, `scripts/harmonize_datasets.py`) is **not required to reproduce our results** — it's kept in the repo to document how the published datasets were built. To replicate training/evaluation, it's enough to point `configs/config.yaml` at the published Hugging Face datasets above.

### TrashNet

- Source: Kaggle — TrashNet by Feyza Ozkefe
- Link: https://www.kaggle.com/datasets/feyzazkefe/trashnet
- Notes: Contains six TrashNet-style classes: cardboard, glass, metal, paper, plastic, and trash.
- License: MIT

### Household Garbage Dataset
- Source: Household Garbage / HGI-style dataset
- Link: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification/data
- Notes: Contains household garbage classes with `default` and `real_world` image contexts.
- Used for: Additional training data and real-world robustness evaluation.
- License: MIT

### The Garbage Dataset (GD)
- Source: The Garbage Dataset / GD
- Link: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- Notes: We use only the `original` folder and ignore `standardized_256` and `standardized_384` to avoid duplicate images.
- Used for: Additional training data after label harmonisation.
- License: MIT

### Tricascade
- Source: Kaggle — `salam035/waste-datasets` (`Dataset_ThirdStage/Dataset`)
- Link: https://www.kaggle.com/datasets/salam035/waste-datasets
- Notes: Contributes the largest share of harmonised images. Several fine-grained categories (e.g. batteries, cigarette butts, smartphones, spray cans) fall outside the six-class label space and are excluded during harmonisation.
- Used for: Additional training data after label harmonisation.

## Data & Training Pipeline

Steps 1–3 rebuild the published Hugging Face datasets from scratch and are only needed if you want to regenerate or extend them; to just reproduce training/evaluation, start at step 4 using the published datasets linked above.

1. **Download** raw datasets — [scripts/download_datasets.py](scripts/download_datasets.py)
2. **Harmonise** labels into the shared six-class scheme — [scripts/harmonize_datasets.py](scripts/harmonize_datasets.py), documented in [docs/dataset_harmonization.md](docs/dataset_harmonization.md)
3. **Publish** the harmonised dataset to Hugging Face — [scripts/create_hf_dataset.py](scripts/create_hf_dataset.py)
4. **Train / evaluate** models through the interactive CLI — [main.py](main.py), see [Running main.py](#running-mainpy) below
5. **Explain** predictions — [scripts/grad_cam.py](scripts/grad_cam.py), [scripts/integrated_gradients.py](scripts/integrated_gradients.py), [scripts/occlusion.py](scripts/occlusion.py), backed by [src/xai.py](src/xai.py)

## Running main.py

`main.py` starts an interactive shell (`(main)` prompt) rather than taking CLI flags directly. Model checkpoints can be pulled from Weights & Biases or loaded from the local `models/` folder:

- `train` — optionally seeds RNGs for reproducibility, then prompts you to pick one or more models from the registry (`resnet18`, `mlp_mixer`, `convnext_tiny`, `deit`) to fine-tune, using the hyperparameters in `configs/config.yaml`
- `test` — prompts you to pick a fine-tuned checkpoint from `models/` and a path to a single image, and prints the predicted class
- `test_custom` — prompts you to pick a fine-tuned checkpoint and evaluates it against the custom domain-shift dataset (`TrashBusters/wild-trash-inference-dataset`), reporting loss, accuracy, top-2 accuracy, and macro-F1
- `exit` (or Ctrl-D) — quits the shell

## Models

Four architectures are registered in `src/models` and share a common training engine ([src/engine.py](src/engine.py)) — AdamW, cross-entropy loss, and a plateau LR scheduler configured in [configs/config.yaml](configs/config.yaml):

- **ResNet-18** — convolutional baseline
- **MLP-Mixer**
- **ConvNeXt-Tiny**
- **DeiT** (Data-efficient image Transformer)

Fine-tuned checkpoints are stored under `models/`.

## Explainability (XAI)

Model predictions can be inspected with three attribution methods implemented in [src/xai.py](src/xai.py):

- **Grad-CAM** — class-discriminative activation heatmaps from the last convolutional/block layer
- **Integrated Gradients** (via Captum) — pixel-level attribution along a path from a baseline to the input
- **Occlusion** (via Captum) — attribution by measuring the prediction change under systematic patch masking

## Evaluation & Notebooks

`notebooks/` contains the analysis used to produce the results below, backed by shared helpers in [notebooks/eval_utils.py](notebooks/eval_utils.py):

- `01_metrics_and_comparison.ipynb` — cross-model metric comparison
- `02_class_analysis.ipynb` — per-class performance breakdown
- `03_trashbusters_dataset_charts.ipynb` — dataset composition charts

## Results

Models are evaluated on an in-distribution held-out split of the harmonised dataset, and separately on a small custom, manually-collected set used to probe generalisation under domain shift.

**All six classes:**

| Architecture | In-Dist. Acc | In-Dist. F1 (macro) | Custom Acc | Custom F1 (macro) | Δ Acc | Δ F1 (macro) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 (baseline) | 0.7950 | 0.7946 | 0.2902 | 0.2737 | -0.5048 | -0.5209 |
| MLP-Mixer | 0.9319 | 0.9319 | 0.3276 | 0.3759 | -0.6043 | -0.5560 |
| DeiT | 0.8964 | 0.8963 | 0.3448 | 0.3758 | -0.5516 | -0.5205 |
| ConvNeXt-Tiny | **0.9455** | **0.9455** | **0.4080** | **0.4529** | -0.5375 | -0.4926 |

**Ablation — `trash` class excluded:**

| Architecture | In-Dist. Acc | In-Dist. Top-2 Acc | In-Dist. F1 (macro) | Custom Acc | Custom Top-2 Acc | Custom F1 (macro) | Δ Acc | Δ Top-2 Acc | Δ F1 (macro) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 (baseline) | 0.8092 | 0.9367 | 0.8088 | 0.4472 | 0.7609 | 0.4097 | -0.3620 | -0.1758 | -0.3991 |
| ConvNeXt-Tiny | **0.9519** | **0.9909** | **0.9519** | **0.5279** | **0.8075** | **0.5485** | -0.4240 | -0.1834 | -0.4034 |

ConvNeXt-Tiny is the strongest architecture in both settings, in-distribution and under domain shift; all models show a substantial accuracy drop on the custom set, and removing the ambiguous `trash` class narrows but does not close that gap.

## Configuration & Experiment Tracking

[configs/config.yaml](configs/config.yaml) controls the Hugging Face dataset repo id, training hyperparameters, the LR scheduler, and the Weights & Biases project/entity. Copy `.env.example` to `.env` and set `WANDB_API_KEY` and `HF_TOKEN` to enable experiment tracking and Hugging Face dataset access.
