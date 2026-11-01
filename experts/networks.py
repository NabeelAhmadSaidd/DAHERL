"""
Shared neural network building blocks for all expert agents.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type


def get_activation(name: str) -> Type[nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(activations.keys())}")
    return activations[name]


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_norm: bool = False,
) -> nn.Sequential:
    """
    Build a fully-connected MLP.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        hidden_dims: List of hidden layer sizes.
        activation: Hidden layer activation name.
        output_activation: Optional output layer activation name.
        layer_norm: If True, apply LayerNorm after each hidden activation.

    Returns:
        nn.Sequential MLP module.
    """
    act_cls = get_activation(activation)
    layers = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(act_cls())
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(get_activation(output_activation)())

    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)