"""Healthcare Rainbow DQN components for advanced temporal difference learning.

This module provides specialized Rainbow DQN network components including noisy
layers, dueling architectures, and distributional RL support, optimized for
healthcare patient care decision-making in temporal difference learning settings.
"""

import math
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typeguard import typechecked

from core.constants import CLINICAL_FEATURE_COUNT, RAINBOW_CONFIG


class NoisyLinear(nn.Module):
    """Noisy linear layer for parameter space noise exploration.

    Implements factorized Gaussian noise for exploration in healthcare DQN,
    replacing epsilon-greedy exploration with learnable noise parameters.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        std_init (float): Initial standard deviation for noise parameters.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize learnable noise parameters.

        Initializes mu and sigma parameters uniformly based on input dimensions
        and the configured std_init value.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Generate new factorized Gaussian noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled factorized Gaussian noise vector.

        Args:
            size (int): Dimension size for the noise vector.

        Returns:
            torch.Tensor: Noise tensor of shape (size,).
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply noisy linear transformation to input.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, out_features).
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class DuelingHealthcareDQN(nn.Module):
    """Dueling Deep Q-Network for healthcare with Rainbow extensions.

    Implements a dueling network architecture with optional noisy layers and
    C51 distributional support for temporal healthcare decision-making.

    Args:
        number_of_actions (int): Number of possible healthcare actions.
        input_dim (int): Dimensionality of clinical feature input.
        use_noisy (bool): Enable noisy parameter layers.
        use_distributional (bool): Enable distributional RL (C51).
    """

    def __init__(
        self,
        number_of_actions: int,
        input_dim: int = CLINICAL_FEATURE_COUNT,
        use_noisy: bool = True,
        use_distributional: bool = True,
    ) -> None:
        """Initialize the dueling healthcare DQN model.

        Args:
            number_of_actions (int): Number of healthcare actions.
            input_dim (int): Number of clinical input features.
            use_noisy (bool): Use NoisyLinear layers for exploration.
            use_distributional (bool): Use C51 distributional RL support.
        """
        super(DuelingHealthcareDQN, self).__init__()
        self.input_dim = input_dim
        self.number_of_actions = number_of_actions
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.num_atoms = RAINBOW_CONFIG["num_atoms"] if use_distributional else 1

        # Shared feature extraction layers
        self.feature_layer1 = nn.Linear(input_dim, 64)
        self.feature_layer2 = nn.Linear(64, 128)
        self.feature_layer3 = nn.Linear(128, 256)

        # Value stream
        if use_noisy:
            self.value_layer1 = NoisyLinear(256, 128)
            self.value_layer2 = NoisyLinear(128, self.num_atoms)
        else:
            self.value_layer1 = nn.Linear(256, 128)
            self.value_layer2 = nn.Linear(128, self.num_atoms)

        # Advantage stream
        if use_noisy:
            self.advantage_layer1 = NoisyLinear(256, 128)
            self.advantage_layer2 = NoisyLinear(128, number_of_actions * self.num_atoms)
        else:
            self.advantage_layer1 = nn.Linear(256, 128)
            self.advantage_layer2 = nn.Linear(128, number_of_actions * self.num_atoms)

        # Distributional RL support
        if use_distributional:
            self.register_buffer(
                "support",
                torch.linspace(
                    RAINBOW_CONFIG["v_min"], RAINBOW_CONFIG["v_max"], self.num_atoms
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through the dueling network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: If distributional, tensor of shape
                (batch_size, number_of_actions, num_atoms); otherwise
                tensor of shape (batch_size, number_of_actions).
        """
        batch_size = x.size(0)

        # Shared feature extraction
        features = F.relu(self.feature_layer1(x))
        features = F.relu(self.feature_layer2(features))
        features = F.relu(self.feature_layer3(features))

        # Value stream
        value = F.relu(self.value_layer1(features))
        value = self.value_layer2(value)

        # Advantage stream
        advantage = F.relu(self.advantage_layer1(features))
        advantage = self.advantage_layer2(advantage)

        if self.use_distributional:
            # Reshape for distributional RL
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(
                batch_size, self.number_of_actions, self.num_atoms
            )

            # Dueling aggregation
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

            # Apply softmax to get probability distributions
            q_dist = F.softmax(q_atoms, dim=-1)
            return q_dist
        else:
            # Standard dueling aggregation
            value = value.view(batch_size, 1)
            advantage = advantage.view(batch_size, self.number_of_actions)

            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values

    def reset_noise(self) -> None:
        """Reset noise parameters in all NoisyLinear layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve Q-values from the network output, aggregating distributions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Expected Q-values tensor of shape (batch_size, number_of_actions).
        """
        if self.use_distributional:
            q_dist = self.forward(x)
            q_values = torch.sum(q_dist * self.support, dim=-1)
            return q_values
        else:
            return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict Q-values for numpy input array.

        Args:
            x (np.ndarray): Input features array of shape (batch_size, input_dim).

        Returns:
            np.ndarray: Predicted Q-values array of shape (batch_size, number_of_actions).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor(x)
            else:
                x_tensor = x

            if self.use_distributional:
                q_values = self.get_q_values(x_tensor)
            else:
                q_values = self.forward(x_tensor)

            return q_values.numpy()


@typechecked
def create_dueling_healthcare_dqn(
    number_of_actions: int,
    input_dim: int = CLINICAL_FEATURE_COUNT,
    use_noisy: bool = True,
    use_distributional: bool = True,
) -> DuelingHealthcareDQN:
    """Instantiate a DuelingHealthcareDQN model with validation.

    Args:
        number_of_actions (int): Number of healthcare actions (>=2).
        input_dim (int): Number of clinical input features (>=1).
        use_noisy (bool): Enable NoisyLinear exploration layers.
        use_distributional (bool): Enable C51 distributional RL.

    Returns:
        DuelingHealthcareDQN: Configured dueling DQN model.

    Raises:
        ValueError: If number_of_actions < 2 or input_dim < 1.
    """
    if number_of_actions < 2:
        raise ValueError("number_of_actions must be at least 2")
    if input_dim < 1:
        raise ValueError("input_dim must be at least 1")

    return DuelingHealthcareDQN(
        number_of_actions=number_of_actions,
        input_dim=input_dim,
        use_noisy=use_noisy,
        use_distributional=use_distributional,
    )


def compute_distributional_loss(
    q_dist: torch.Tensor, target_dist: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Compute cross-entropy loss for distributional RL (C51).

    Args:
        q_dist (torch.Tensor): Predicted distributions tensor of shape
            (batch_size, num_actions, num_atoms).
        target_dist (torch.Tensor): Target distributions tensor of shape
            (batch_size, num_atoms).
        actions (torch.Tensor): Indices of actions taken of shape (batch_size,).

    Returns:
        torch.Tensor: Scalar loss value averaged over the batch.
    """
    batch_size = q_dist.size(0)

    # Select distributions for taken actions
    q_dist_selected = q_dist[range(batch_size), actions]

    # Compute cross-entropy loss
    loss = -torch.sum(target_dist * torch.log(q_dist_selected + 1e-8), dim=1)
    return loss.mean()
