import torch
from torch import nn

SUPPORTED_ACTIVATION_MAP = {
    "ReLU",
    "Sigmoid",
    "Tanh",
    "ELU",
    "SELU",
    "GLU",
    "LeakyReLU",
    "Softplus",
    "None",
}


def get_activation(activation):
    """returns the activation function represented by the input string"""
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()
    ]
    assert len(activation) == 1 and isinstance(
        activation[0], str
    ), "Unhandled activation function"
    activation = activation[0]
    if activation.lower() == "none":
        return None
    return vars(torch.nn.modules.activation)[activation]()


ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU}
