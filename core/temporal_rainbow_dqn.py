"""Healthcare refactored temporal factored DQN for healthcare communication optimization.

This module implements a sophisticated Rainbow DQN with factored temporal actions,
including healthcare decisions, timing, communication channels, and patient state
encoding, optimized for healthcare reinforcement learning.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.temporal_actions import TemporalActionSpace, TemporalAction
from core.rainbow_components import NoisyLinear
from core.constants import RAINBOW_CONFIG
from core.logging_system import get_logger

logger = get_logger(__name__)


class TemporalFactoredDQN(nn.Module):
    """Factored Rainbow DQN for temporal healthcare communication optimization.

    Uses composable heads per action dimension with support for dueling,
    noisy layers, and distributional C51 RL.

    Args:
        input_dim (int): Size of the enhanced patient state vector.
        use_dueling (bool): Enable dueling streams. Defaults to True.
        use_noisy (bool): Enable NoisyLinear exploration layers. Defaults to True.
        use_distributional (bool): Enable distributional RL support. Defaults to True.
        hidden_dim (int): Hidden layer size. Defaults to 256.
    """

    def __init__(
        self,
        input_dim: int = 27,
        use_dueling: bool = True,
        use_noisy: bool = True,
        use_distributional: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the TemporalFactoredDQN model.

        Sets up shared feature extractor, factored heads, and distributional support.

        Args:
            input_dim (int): Size of the enhanced patient state vector.
            use_dueling (bool): Enable dueling streams.
            use_noisy (bool): Enable NoisyLinear layers.
            use_distributional (bool): Enable distributional RL (C51).
            hidden_dim (int): Hidden layer dimension.
        """
        super(TemporalFactoredDQN, self).__init__()

        self.input_dim = input_dim
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.hidden_dim = hidden_dim

        # Action space dimensions
        self.action_space = TemporalActionSpace()
        self.n_healthcare = self.action_space.n_healthcare
        self.n_time_horizons = self.action_space.n_time_horizons
        self.n_times_of_day = self.action_space.n_times_of_day
        self.n_communication = self.action_space.n_communication

        # Distributional RL setup
        if use_distributional:
            self.num_atoms = RAINBOW_CONFIG["num_atoms"]
            self.register_buffer(
                "support",
                torch.linspace(
                    RAINBOW_CONFIG["v_min"], RAINBOW_CONFIG["v_max"], self.num_atoms
                ),
            )
        else:
            self.num_atoms = 1

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Create factored heads
        self._create_factored_heads()

        # Log initialization using structured logging
        logger.log_network_initialization(
            "TemporalFactoredDQN",
            {
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "healthcare_actions": self.n_healthcare,
                "time_horizons": self.n_time_horizons,
                "times_of_day": self.n_times_of_day,
                "communication_channels": self.n_communication,
                "dueling": use_dueling,
                "noisy": use_noisy,
                "distributional": use_distributional,
                "atoms_per_action": self.num_atoms if use_distributional else 1,
            },
        )

    def _create_factored_heads(self) -> None:
        """Create and initialize factored heads for each action dimension."""
        if self.use_dueling:
            # Dueling architecture: separate value and advantage streams
            if self.use_noisy:
                self.healthcare_value = NoisyLinear(self.hidden_dim, self.num_atoms)
                self.timing_value = NoisyLinear(self.hidden_dim, self.num_atoms)
                self.schedule_value = NoisyLinear(self.hidden_dim, self.num_atoms)
                self.communication_value = NoisyLinear(self.hidden_dim, self.num_atoms)
            else:
                self.healthcare_value = nn.Linear(self.hidden_dim, self.num_atoms)
                self.timing_value = nn.Linear(self.hidden_dim, self.num_atoms)
                self.schedule_value = nn.Linear(self.hidden_dim, self.num_atoms)
                self.communication_value = nn.Linear(self.hidden_dim, self.num_atoms)

            if self.use_noisy:
                self.healthcare_advantage = NoisyLinear(
                    self.hidden_dim, self.n_healthcare * self.num_atoms
                )
                self.timing_advantage = NoisyLinear(
                    self.hidden_dim, self.n_time_horizons * self.num_atoms
                )
                self.schedule_advantage = NoisyLinear(
                    self.hidden_dim, self.n_times_of_day * self.num_atoms
                )
                self.communication_advantage = NoisyLinear(
                    self.hidden_dim, self.n_communication * self.num_atoms
                )
            else:
                self.healthcare_advantage = nn.Linear(
                    self.hidden_dim, self.n_healthcare * self.num_atoms
                )
                self.timing_advantage = nn.Linear(
                    self.hidden_dim, self.n_time_horizons * self.num_atoms
                )
                self.schedule_advantage = nn.Linear(
                    self.hidden_dim, self.n_times_of_day * self.num_atoms
                )
                self.communication_advantage = nn.Linear(
                    self.hidden_dim, self.n_communication * self.num_atoms
                )
        else:
            if self.use_noisy:
                self.healthcare_head = NoisyLinear(
                    self.hidden_dim, self.n_healthcare * self.num_atoms
                )
                self.timing_head = NoisyLinear(
                    self.hidden_dim, self.n_time_horizons * self.num_atoms
                )
                self.schedule_head = NoisyLinear(
                    self.hidden_dim, self.n_times_of_day * self.num_atoms
                )
                self.communication_head = NoisyLinear(
                    self.hidden_dim, self.n_communication * self.num_atoms
                )
            else:
                self.healthcare_head = nn.Linear(
                    self.hidden_dim, self.n_healthcare * self.num_atoms
                )
                self.timing_head = nn.Linear(
                    self.hidden_dim, self.n_time_horizons * self.num_atoms
                )
                self.schedule_head = nn.Linear(
                    self.hidden_dim, self.n_times_of_day * self.num_atoms
                )
                self.communication_head = nn.Linear(
                    self.hidden_dim, self.n_communication * self.num_atoms
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform forward pass through factored network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Dict[str, torch.Tensor]: Q-values or distributions per dimension.
        """
        batch_size = x.size(0)
        features = self.shared_layers(x)
        if self.use_dueling:
            outputs = {}
            healthcare_value = self.healthcare_value(features)
            healthcare_advantage = self.healthcare_advantage(features)
            outputs["healthcare"] = self._compute_dueling_q_values(
                healthcare_value, healthcare_advantage, batch_size, self.n_healthcare
            )
            timing_value = self.timing_value(features)
            timing_advantage = self.timing_advantage(features)
            outputs["timing"] = self._compute_dueling_q_values(
                timing_value, timing_advantage, batch_size, self.n_time_horizons
            )
            schedule_value = self.schedule_value(features)
            schedule_advantage = self.schedule_advantage(features)
            outputs["schedule"] = self._compute_dueling_q_values(
                schedule_value, schedule_advantage, batch_size, self.n_times_of_day
            )
            communication_value = self.communication_value(features)
            communication_advantage = self.communication_advantage(features)
            outputs["communication"] = self._compute_dueling_q_values(
                communication_value,
                communication_advantage,
                batch_size,
                self.n_communication,
            )
        else:
            outputs = {
                "healthcare": self._process_head_output(
                    self.healthcare_head(features), batch_size, self.n_healthcare
                ),
                "timing": self._process_head_output(
                    self.timing_head(features), batch_size, self.n_time_horizons
                ),
                "schedule": self._process_head_output(
                    self.schedule_head(features), batch_size, self.n_times_of_day
                ),
                "communication": self._process_head_output(
                    self.communication_head(features), batch_size, self.n_communication
                ),
            }
        return outputs

    def _compute_dueling_q_values(
        self,
        value: torch.Tensor,
        advantage: torch.Tensor,
        batch_size: int,
        n_actions: int,
    ) -> torch.Tensor:
        """Compute dueling Q-values or distributions for one dimension.

        Args:
            value (torch.Tensor): Value stream output.
            advantage (torch.Tensor): Advantage stream output.
            batch_size (int): Batch size.
            n_actions (int): Number of actions in this dimension.

        Returns:
            torch.Tensor: Dueling Q-values or probability distributions.
        """
        if self.use_distributional:
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, n_actions, self.num_atoms)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
            q_dist = F.softmax(q_atoms, dim=-1)
            return q_dist
        else:
            value = value.view(batch_size, 1)
            advantage = advantage.view(batch_size, n_actions)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values

    def _process_head_output(
        self, output: torch.Tensor, batch_size: int, n_actions: int
    ) -> torch.Tensor:
        """Process a standard head output into Q-values or distributions.

        Args:
            output (torch.Tensor): Head output tensor.
            batch_size (int): Batch size.
            n_actions (int): Number of actions.

        Returns:
            torch.Tensor: Q-values or probability distributions.
        """
        if self.use_distributional:
            output = output.view(batch_size, n_actions, self.num_atoms)
            q_dist = F.softmax(output, dim=-1)
            return q_dist
        else:
            return output.view(batch_size, n_actions)

    def get_factored_q_values(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get per-dimension Q-values or distributions from network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Dict[str, torch.Tensor]: Factored outputs per action dimension.
        """
        outputs = self.forward(x)
        if self.use_distributional:
            q_values = {}
            for dimension, q_dist in outputs.items():
                q_values[dimension] = torch.sum(q_dist * self.support, dim=-1)
            return q_values
        else:
            return outputs

    def combine_factored_q_values(
        self,
        factored_q_values: Dict[str, torch.Tensor],
        combination_method: str = "additive",
    ) -> torch.Tensor:
        """Combine factored Q-values into joint action Q-values.

        Args:
            factored_q_values (Dict[str, torch.Tensor]): Q-values per dimension.
            combination_method (str): "additive" or "multiplicative".

        Returns:
            torch.Tensor: Combined Q-values tensor.

        Raises:
            ValueError: If combination_method is not recognized.
        """
        batch_size = factored_q_values["healthcare"].size(0)
        healthcare_q = factored_q_values["healthcare"]
        timing_q = factored_q_values["timing"]
        schedule_q = factored_q_values["schedule"]
        communication_q = factored_q_values["communication"]
        if combination_method == "additive":
            combined_q = (
                healthcare_q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                + timing_q.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                + schedule_q.unsqueeze(1).unsqueeze(2).unsqueeze(4)
                + communication_q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            )
            combined_q = combined_q.view(batch_size, -1)
        elif combination_method == "multiplicative":
            healthcare_q = torch.sigmoid(healthcare_q)
            timing_q = torch.sigmoid(timing_q)
            schedule_q = torch.sigmoid(schedule_q)
            communication_q = torch.sigmoid(communication_q)
            combined_q = (
                healthcare_q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                * timing_q.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                * schedule_q.unsqueeze(1).unsqueeze(2).unsqueeze(4)
                * communication_q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            )
            combined_q = combined_q.view(batch_size, -1)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        return combined_q

    def predict_temporal_action(
        self,
        x: np.ndarray,
        use_exploration: bool = True,
        temperature: float = 1.2,
        epsilon: float = 0.1,
    ) -> TemporalAction:
        """Predict a temporal action with optional exploration.

        Args:
            x (np.ndarray): Input feature array.
            use_exploration (bool): Enable exploration.
            temperature (float): Softmax temperature.
            epsilon (float): Epsilon for random action.

        Returns:
            TemporalAction: Selected composite action.

        Raises:
            TypeError: If x is not a numpy array or torch.Tensor.
        """
        if use_exploration:
            self.train()
        else:
            self.eval()
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device

            if isinstance(x, np.ndarray):
                x_np = np.array(x) if x.ndim == 1 else x
                x_tensor = torch.FloatTensor(x_np.reshape(1, -1)).to(device)
            else:
                x_tensor = x.to(device)
            if use_exploration and np.random.random() < epsilon:
                return self.action_space.sample_random_action()
            factored_q_values = self.get_factored_q_values(x_tensor)
            if use_exploration and temperature > 0:
                healthcare_probs = torch.softmax(
                    factored_q_values["healthcare"][0] / temperature, dim=0
                )
                timing_probs = torch.softmax(
                    factored_q_values["timing"][0] / temperature, dim=0
                )
                schedule_probs = torch.softmax(
                    factored_q_values["schedule"][0] / temperature, dim=0
                )
                communication_probs = torch.softmax(
                    factored_q_values["communication"][0] / temperature, dim=0
                )
                healthcare_idx = torch.multinomial(healthcare_probs, 1).item()
                timing_idx = torch.multinomial(timing_probs, 1).item()
                schedule_idx = torch.multinomial(schedule_probs, 1).item()
                communication_idx = torch.multinomial(communication_probs, 1).item()
            else:
                healthcare_idx = torch.argmax(factored_q_values["healthcare"][0]).item()
                timing_idx = torch.argmax(factored_q_values["timing"][0]).item()
                schedule_idx = torch.argmax(factored_q_values["schedule"][0]).item()
                communication_idx = torch.argmax(
                    factored_q_values["communication"][0]
                ).item()
            return self.action_space.indices_to_action(
                healthcare_idx, timing_idx, schedule_idx, communication_idx
            )

    def get_action_probabilities(
        self, x: np.ndarray, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Get softmax probability distributions per dimension.

        Args:
            x (np.ndarray): Input feature array.
            temperature (float): Softmax temperature.

        Returns:
            Dict[str, torch.Tensor]: Probability distributions per dimension.
        """
        self.eval()
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device

            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor([x]).to(device)
            else:
                x_tensor = x.to(device)
            factored_q_values = self.get_factored_q_values(x_tensor)
            probabilities = {}
            for dimension, q_values in factored_q_values.items():
                probabilities[dimension] = torch.softmax(
                    q_values[0] / temperature, dim=0
                )
            return probabilities

    def reset_noise(self) -> None:
        """Reset noise parameters in all NoisyLinear layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


def create_temporal_factored_dqn(
    input_dim: int = 27,
    use_dueling: bool = True,
    use_noisy: bool = True,
    use_distributional: bool = True,
    hidden_dim: int = 256,
) -> TemporalFactoredDQN:
    """Factory to create a TemporalFactoredDQN instance.

    Args:
        input_dim (int): Enhanced state vector size.
        use_dueling (bool): Enable dueling streams.
        use_noisy (bool): Enable NoisyLinear layers.
        use_distributional (bool): Enable distributional RL.
        hidden_dim (int): Hidden layer size.

    Returns:
        TemporalFactoredDQN: Configured model.
    """
    return TemporalFactoredDQN(
        input_dim=input_dim,
        use_dueling=use_dueling,
        use_noisy=use_noisy,
        use_distributional=use_distributional,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    network = create_temporal_factored_dqn()
    logger.info(
        "Temporal factored DQN test initialization completed",
        parameter_count=sum(p.numel() for p in network.parameters()),
    )
