'''
Loads individual model factory functions and exposes a name -> function
registry (MODEL_REGISTRY) plus the default selection used for training.
'''
from .convnext_tiny import convnext_tiny
from .deit import deit
from .resnet import resnet18
from .mlp_mixer import mlp_mixer


# Name -> factory function. Select models to train by these names.
MODEL_REGISTRY = {
    "resnet18": resnet18,
    "mlp_mixer": mlp_mixer,
    "convnext_tiny": convnext_tiny,
    "deit": deit,
}

# Default selection when train_all() is called without explicit model names.
DEFAULT_MODEL_NAMES = list(MODEL_REGISTRY)
