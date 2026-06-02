# Project

Project scaffold for data, notebooks, source code, configuration, and outputs.

## Conda Environment

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate base-project
```

Update it after changing dependencies:

```bash
conda env update -f environment.yml --prune
```
