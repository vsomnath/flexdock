import torch.nn as nn

from flexdock.models.layers.activation import ACTIVATIONS


def FCBlock(in_dim, hidden_dim, out_dim, layers, dropout, activation="relu"):
    activation = ACTIVATIONS[activation]
    assert layers >= 2
    sequential = [nn.Linear(in_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    for i in range(layers - 2):
        sequential += [
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
        ]
    sequential += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*sequential)
