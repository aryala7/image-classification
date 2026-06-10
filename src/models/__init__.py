'''
Loads individual models as functions,
also gives over MODELS list which contains tuples of (model_fn, epochs) for training.
'''
from .resnet import resnet18
from .mlp_mixer import mlp_mixer



MODELS = [(resnet18, 5), (mlp_mixer, 5)]
