# Project

Project scaffold for data, notebooks, source code, configuration, and outputs.

## Conda Environment

Create the environment:

```bash
conda env create -f environment.yml
```

Update the environment, **depending on cuda support**:

```bash
conda env update -f environment.<cuda/cpu>.yml
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

This project combines and harmonises the following public waste-classification datasets:

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
