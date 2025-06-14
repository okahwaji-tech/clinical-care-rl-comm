"""Healthcare prioritized experience replay buffer for temporal difference learning.

This module implements a prioritized experience replay buffer with a sum tree
data structure for efficient sampling based on temporal difference errors
in healthcare decision-making scenarios.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from typeguard import typechecked

from core.constants import PER_CONFIG


@dataclass
class HealthcareExperience:
    """Healthcare experience tuple for replay buffer.

    Represents a single healthcare decision experience including patient state,
    action taken, reward received, and next state.

    Attributes:
        state (np.ndarray): Patient clinical features at time t.
        action (int): Healthcare action taken (index).
        reward (float): Clinical outcome reward.
        next_state (np.ndarray): Patient clinical features at time t+1.
        done (bool): Whether episode terminated.
        n_step_return (Optional[float]): N-step return for multi-step learning.
        n_step_state (Optional[np.ndarray]): State after n steps.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    n_step_return: Optional[float] = None
    n_step_state: Optional[np.ndarray] = None


class SumTree:
    """Sum tree data structure for efficient prioritized sampling.

    Implements a binary tree where each leaf contains a priority value and each
    internal node contains the sum of its children's values. Enables O(log n)
    sampling and updating operations.

    Args:
        capacity (int): Maximum number of experiences to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree.

        Args:
            idx (int): Index of the leaf node.
            change (float): Amount to add to the leaf's value.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for a given cumulative sum.

        Args:
            idx (int): Current tree index.
            s (float): Cumulative sum to locate.

        Returns:
            int: Leaf index corresponding to the sum.
        """
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get total sum of all priorities.

        Returns:
            float: Sum of all priority values.
        """
        return self.tree[0]

    def add(self, priority: float, data: HealthcareExperience) -> None:
        """Add new experience with given priority.

        Args:
            priority (float): Priority value for this experience.
            data (HealthcareExperience): Experience to add.
        """
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority of an experience at a given index.

        Args:
            idx (int): Index in the tree to update.
            priority (float): New priority value.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, HealthcareExperience]:
        """Get experience for a given cumulative sum.

        Args:
            s (float): Cumulative sum to sample.

        Returns:
            Tuple[int, float, HealthcareExperience]: (tree index, priority, experience).
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


@typechecked
class PrioritizedReplayBuffer:
    """Initialize the prioritized replay buffer.

    Args:
        capacity (int): Maximum number of experiences to store.
        alpha (float): Prioritization exponent (0 = uniform, 1 = full prioritization).
        beta_start (float): Initial importance sampling weight.
        beta_frames (int): Number of frames to anneal beta to 1.0.
        n_step (int): Number of steps for n-step returns.
        gamma (float): Discount factor for n-step returns.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = PER_CONFIG["alpha"],
        beta_start: float = PER_CONFIG["beta_start"],
        beta_frames: int = PER_CONFIG["beta_frames"],
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.n_step = n_step
        self.gamma = gamma

        self.tree = SumTree(capacity)
        self.max_priority = PER_CONFIG["max_priority"]
        self.frame_idx = 0

        # N-step buffer for multi-step returns
        self.n_step_buffer: List[HealthcareExperience] = []

    def _get_beta(self) -> float:
        """Get current beta value with annealing.

        Returns:
            float: Annealed beta value.
        """
        return min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * self.frame_idx / self.beta_frames,
        )

    def _compute_n_step_return(
        self, experiences: List[HealthcareExperience]
    ) -> Tuple[float, np.ndarray, bool]:
        """Compute n-step return from a sequence of experiences.

        Args:
            experiences (List[HealthcareExperience]): List of experiences.

        Returns:
            Tuple[float, np.ndarray, bool]: n-step return, next state, and done flag.
        """
        n_step_return = 0.0
        gamma_power = 1.0
        for i, exp in enumerate(experiences):
            n_step_return += gamma_power * exp.reward
            gamma_power *= self.gamma
            if exp.done:
                return n_step_return, experiences[i].next_state, True
        # If no terminal state reached, return final state
        return n_step_return, experiences[-1].next_state, False

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new experience to the replay buffer with n-step processing.

        Args:
            state (np.ndarray): Patient clinical features at time t.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state features.
            done (bool): Episode termination flag.
        """
        experience = HealthcareExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=bool(done),  # Convert numpy bool to Python bool
        )
        self.n_step_buffer.append(experience)
        # Process n-step return when buffer is full or episode ends
        if len(self.n_step_buffer) >= self.n_step or done:
            if len(self.n_step_buffer) >= self.n_step:
                # Compute n-step return
                n_step_return, n_step_state, n_step_done = self._compute_n_step_return(
                    self.n_step_buffer[: self.n_step]
                )
                # Update first experience with n-step information
                first_exp = self.n_step_buffer[0]
                first_exp.n_step_return = n_step_return
                first_exp.n_step_state = n_step_state
                # Add to tree with maximum priority
                self.tree.add(self.max_priority**self.alpha, first_exp)
                # Remove processed experience
                self.n_step_buffer.pop(0)
            # If episode ended, process remaining experiences
            if done:
                while self.n_step_buffer:
                    remaining_steps = len(self.n_step_buffer)
                    (
                        n_step_return,
                        n_step_state,
                        n_step_done,
                    ) = self._compute_n_step_return(
                        self.n_step_buffer[:remaining_steps]
                    )
                    exp = self.n_step_buffer.pop(0)
                    exp.n_step_return = n_step_return
                    exp.n_step_state = n_step_state
                    self.tree.add(self.max_priority**self.alpha, exp)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences with importance sampling weights.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            Tuple[torch.Tensor, ...]: states, actions, rewards, next_states, dones,
                n_step_returns, n_step_states, weights, indices.

        Raises:
            ValueError: If buffer size is less than batch_size.
        """
        if self.tree.size < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer: {self.tree.size} < {batch_size}"
            )
        batch_indices = []
        batch_experiences = []
        batch_priorities = []
        # Sample experiences
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            batch_indices.append(idx)
            batch_experiences.append(experience)
            batch_priorities.append(priority)
        # Compute importance sampling weights
        beta = self._get_beta()
        self.frame_idx += 1
        sampling_probabilities = np.array(batch_priorities) / self.tree.total()
        weights = (self.tree.size * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        # Convert to numpy arrays first for efficiency with proper dtype handling
        states_np = np.array([exp.state for exp in batch_experiences], dtype=np.float32)
        actions_np = np.array([exp.action for exp in batch_experiences], dtype=np.int64)
        rewards_np = np.array([exp.reward for exp in batch_experiences], dtype=np.float32)
        next_states_np = np.array([exp.next_state for exp in batch_experiences], dtype=np.float32)
        dones_np = np.array([exp.done for exp in batch_experiences], dtype=bool)
        # N-step returns and states with proper dtype
        n_step_returns_np = np.array(
            [
                exp.n_step_return if exp.n_step_return is not None else exp.reward
                for exp in batch_experiences
            ],
            dtype=np.float32
        )
        n_step_states_np = np.array(
            [
                exp.n_step_state if exp.n_step_state is not None else exp.next_state
                for exp in batch_experiences
            ],
            dtype=np.float32
        )
        # Convert numpy arrays to tensors (much faster)
        states = torch.FloatTensor(states_np)
        actions = torch.LongTensor(actions_np)
        rewards = torch.FloatTensor(rewards_np)
        next_states = torch.FloatTensor(next_states_np)
        dones = torch.BoolTensor(dones_np)
        n_step_returns = torch.FloatTensor(n_step_returns_np)
        n_step_states = torch.FloatTensor(n_step_states_np)
        weights_tensor = torch.FloatTensor(weights)
        indices_tensor = torch.LongTensor(batch_indices)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            n_step_returns,
            n_step_states,
            weights_tensor,
            indices_tensor,
        )

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor, states: torch.Tensor = None, actions: torch.Tensor = None
    ) -> None:
        """Update priorities for a set of experiences with risk-aware adjustment.

        Args:
            indices (torch.Tensor): Tensor of tree indices to update.
            priorities (torch.Tensor): Corresponding new priorities (TD errors).
            states (torch.Tensor, optional): State tensors for risk-aware adjustment.
            actions (torch.Tensor, optional): Action tensors for risk-aware adjustment.
        """
        for i, (idx, priority) in enumerate(zip(indices.cpu().numpy(), priorities.cpu().numpy())):
            # Base priority calculation
            base_priority = abs(priority) + PER_CONFIG["epsilon"]

            # Risk-aware priority adjustment
            if states is not None and actions is not None:
                adjusted_priority = self._apply_risk_aware_adjustment(
                    base_priority, states[i], actions[i]
                )
            else:
                adjusted_priority = base_priority

            self.tree.update(idx, adjusted_priority**self.alpha)
            self.max_priority = max(self.max_priority, adjusted_priority)

    def _apply_risk_aware_adjustment(
        self, base_priority: float, state: torch.Tensor, action: torch.Tensor
    ) -> float:
        """Apply risk-aware priority adjustment.

        Multiplies priority by 1.5 for transitions where risk_score >= 0.7
        and action is REFER (1) or MEDICATE (3).

        Args:
            base_priority (float): Base TD error priority
            state (torch.Tensor): Patient state tensor
            action (int): Healthcare action taken

        Returns:
            float: Adjusted priority value
        """
        # Extract risk score from state (assuming it's the second feature)
        # This assumes the state vector follows the EnhancedPatientState feature order
        if len(state.shape) == 0:
            # Handle scalar case
            risk_score = 0.5  # Default fallback
        else:
            # Risk score is typically the second feature in our state representation
            risk_score = float(state[1]) if state.numel() > 1 else 0.5

        # Decode healthcare action from composite action
        # Action encoding: healthcare_action * 8 * 4 * 4 + timing * 4 * 4 + schedule * 4 + communication
        action_value = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        healthcare_action = action_value // (8 * 4 * 4)  # Extract healthcare action

        # Apply 1.5x multiplier for high-risk REFER/MEDICATE transitions
        if risk_score >= 0.7 and healthcare_action in [1, 3]:  # REFER=1, MEDICATE=3
            return base_priority * 1.5

        return base_priority

    def __len__(self) -> int:
        """Get current size of the replay buffer.

        Returns:
            int: Number of stored experiences.
        """
        return self.tree.size
