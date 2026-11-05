"""Model definition and training.

- ``neighbours``: per-slide k-nearest-neighbour table and normalised coordinates.
- ``program_net``: the network, ``y = f(features) + g(fourier(x, y), slide_embedding)``.
- ``losses``: MSE on program scores, Laplacian smoothness over neighbours, lineage cross-entropy.
- ``train``: neighbourhood-union minibatching, early stopping, prediction.
"""
from .neighbours import knn_table, norm_xy
from .program_net import ProgramNet
from .losses import program_loss
from .train import train_model, predict

__all__ = ["knn_table", "norm_xy", "ProgramNet", "program_loss", "train_model", "predict"]
