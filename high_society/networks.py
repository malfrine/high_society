import torch.nn as nn

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int
) -> nn.Module:
    """Builds a feedforward neural network"""
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(nn.Softplus())
    mlp = nn.Sequential(*layers)
    return mlp