"""Neural-network architecture and parameter initialization."""

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """A small fully connected binary classifier."""

    def __init__(self, number_of_features: int) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(number_of_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized score (logit) per sample."""
        return self.linear_relu_stack(features).squeeze(1)


def initialize_parameters(module: nn.Module) -> None:
    """Initialize linear layers with Xavier-uniform weights and zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
