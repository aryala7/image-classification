"""Run the existing training pipeline for the MLP-Mixer model only, 3 epochs.

Reuses train_all() from src.train (which pulls data from data_test.py and runs
the full pipeline), but swaps the MODELS list so only mlp_mixer trains instead
of every model. Run from the repo root:

    python quick_test_mixer.py
"""

import src.train as train
from src.models.mlp_mixer import mlp_mixer

# Override the module-level MODELS so train_all() only runs this one model.
train.MODELS = [(mlp_mixer, 3)]

if __name__ == "__main__":
    train.train_all()
