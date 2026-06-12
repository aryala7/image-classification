"""Per-dataset modules.

Each module is self-contained and exposes the dataset-specific bits:
- ``NAME``, ``KAGGLE_ID``, ``KAGGLE_SUBPATH`` for downloading
- ``collect(...)`` to walk the raw folders and build harmonized samples

To add a new dataset: create a module here following the same shape and add
it to ``ALL`` below. No other code needs to change.
"""
from . import trashnet, tricascade, household_garbage, garbage_dataset

ALL = [trashnet, tricascade, household_garbage, garbage_dataset]
