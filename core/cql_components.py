"""Conservative Q-Learning components for healthcare reinforcement learning.

This module implements CQL regularization to prevent overestimation of Q-values
for out-of-distribution actions in healthcare decision-making scenarios.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from typeguard import typechecked

from core.constants import CQL_CONFIG


@typechecked
def compute_cql_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    next_q_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    cql_alpha: float = CQL_CONFIG["cql_alpha"],
    cql_temperature: float = CQL_CONFIG["cql_temperature"],
    num_random_actions: int = CQL_CONFIG["num_random_actions"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Conservative Q-Learning loss with healthcare-specific regularization.

    This function implements CQL regularization that penalizes Q-values for
    out-of-distribution actions while preserving standard Bellman error for
    in-distribution actions.

    Args:
        q_values (torch.Tensor): Current Q-values of shape [batch_size, num_actions].
        actions (torch.Tensor): Actions taken of shape [batch_size].
        next_q_values (torch.Tensor): Next state Q-values of shape [batch_size, num_actions].
        rewards (torch.Tensor): Rewards received of shape [batch_size].
        dones (torch.Tensor): Episode termination flags of shape [batch_size].
        gamma (float): Discount factor. Defaults to 0.99.
        cql_alpha (float): CQL regularization coefficient. Defaults to CQL_CONFIG["cql_alpha"].
        cql_temperature (float): Temperature for CQL logsumexp computation.
            Defaults to CQL_CONFIG["cql_temperature"].
        num_random_actions (int): Number of random actions for CQL penalty.
            Defaults to CQL_CONFIG["num_random_actions"].

    Returns:
        total_loss (torch.Tensor): Combined Bellman and CQL loss.
        bellman_loss (torch.Tensor): Bellman error loss.
        cql_loss (torch.Tensor): CQL regularization loss.
    """
    batch_size, num_actions = q_values.shape
    device = q_values.device

    # Standard Bellman error (TD loss)
    q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_max = next_q_values.max(dim=1)[0]
        target_q_values = rewards + gamma * next_q_max * (~dones)

    bellman_loss = F.mse_loss(q_values_selected, target_q_values)

    # CQL regularization term
    # 1. Compute logsumexp over all actions for current states
    current_state_logsumexp = torch.logsumexp(q_values / cql_temperature, dim=1)

    # 2. Generate random actions for out-of-distribution penalty
    random_actions = torch.randint(
        0, num_actions, (batch_size, num_random_actions), device=device
    )

    # 3. Compute Q-values for random actions
    random_q_values = q_values.unsqueeze(1).expand(-1, num_random_actions, -1)
    random_q_selected = random_q_values.gather(2, random_actions.unsqueeze(2)).squeeze(
        2
    )
    random_logsumexp = torch.logsumexp(random_q_selected / cql_temperature, dim=1)

    # 4. CQL penalty: encourage lower Q-values for out-of-distribution actions
    cql_penalty = current_state_logsumexp + random_logsumexp - 2 * q_values_selected
    cql_loss = cql_penalty.mean()

    # Total loss
    total_loss = bellman_loss + cql_alpha * cql_loss

    return total_loss, bellman_loss, cql_loss


@typechecked
def compute_distributional_cql_loss(
    q_dist: torch.Tensor,
    target_dist: torch.Tensor,
    actions: torch.Tensor,
    support: torch.Tensor,
    cql_alpha: float = CQL_CONFIG["cql_alpha"],
    cql_temperature: float = CQL_CONFIG["cql_temperature"],
    num_random_actions: int = CQL_CONFIG["num_random_actions"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CQL loss for distributional RL (C51) with healthcare regularization.

    Extends CQL to work with distributional Q-learning by computing
    conservative penalties on the expected Q-values derived from
    the learned distributions.

    Args:
        q_dist (torch.Tensor): Predicted Q-value distributions of shape [batch_size, num_actions, num_atoms].
        target_dist (torch.Tensor): Target Q-value distributions of shape [batch_size, num_atoms].
        actions (torch.Tensor): Selected actions of shape [batch_size].
        support (torch.Tensor): Support values for distributional RL of shape [num_atoms].
        cql_alpha (float): CQL regularization coefficient. Defaults to CQL_CONFIG["cql_alpha"].
        cql_temperature (float): Temperature for CQL logsumexp computation.
            Defaults to CQL_CONFIG["cql_temperature"].
        num_random_actions (int): Number of random actions for CQL penalty.
            Defaults to CQL_CONFIG["num_random_actions"].

    Returns:
        total_loss (torch.Tensor): Combined distributional and CQL loss.
        distributional_loss (torch.Tensor): Distributional cross-entropy loss.
        cql_loss (torch.Tensor): CQL regularization loss.
    """
    batch_size, num_actions, _ = q_dist.shape  # num_atoms unused
    device = q_dist.device

    # Standard distributional loss (cross-entropy)
    q_dist_selected = q_dist[range(batch_size), actions]
    distributional_loss = -torch.sum(
        target_dist * torch.log(q_dist_selected + 1e-8), dim=1
    ).mean()

    # Convert distributions to Q-values for CQL computation
    q_values = torch.sum(q_dist * support.unsqueeze(0).unsqueeze(0), dim=-1)
    q_values_selected = q_values[range(batch_size), actions]

    # CQL regularization on expected Q-values
    # 1. Compute logsumexp over all actions
    current_state_logsumexp = torch.logsumexp(q_values / cql_temperature, dim=1)

    # 2. Generate random actions for out-of-distribution penalty
    random_actions = torch.randint(
        0, num_actions, (batch_size, num_random_actions), device=device
    )

    # 3. Compute Q-values for random actions
    random_q_values = q_values.unsqueeze(1).expand(-1, num_random_actions, -1)
    random_q_selected = random_q_values.gather(2, random_actions.unsqueeze(2)).squeeze(
        2
    )
    random_logsumexp = torch.logsumexp(random_q_selected / cql_temperature, dim=1)

    # 4. CQL penalty for distributional case
    cql_penalty = current_state_logsumexp + random_logsumexp - 2 * q_values_selected
    cql_loss = cql_penalty.mean()

    # Total loss
    total_loss = distributional_loss + cql_alpha * cql_loss

    return total_loss, distributional_loss, cql_loss


@typechecked
def compute_healthcare_action_penalty(
    q_values: torch.Tensor,
    valid_actions_mask: torch.Tensor,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute healthcare-specific action penalty for invalid clinical actions.

    Applies additional penalty to Q-values for actions that are clinically
    inappropriate or contraindicated based on patient state or medical guidelines.

    Args:
        q_values (torch.Tensor): Current Q-values of shape [batch_size, num_actions].
        valid_actions_mask (torch.Tensor): Binary mask indicating valid actions of shape [batch_size, num_actions].
        penalty_weight (float): Weight for the penalty term. Defaults to 1.0.

    Returns:
        torch.Tensor: Penalty loss for invalid healthcare actions.
    """
    # Penalty for invalid actions (where mask is 0)
    invalid_actions_mask = ~valid_actions_mask
    invalid_q_values = q_values * invalid_actions_mask.float()

    # Encourage lower Q-values for invalid actions
    penalty = torch.sum(torch.relu(invalid_q_values), dim=1).mean()

    return penalty_weight * penalty


@typechecked
def compute_safety_regularization(
    q_values: torch.Tensor,
    safety_scores: torch.Tensor,
    safety_threshold: float = 0.5,
    safety_weight: float = 1.0,
) -> torch.Tensor:
    """Compute safety regularization for healthcare actions.

    Penalizes Q-values for actions with low safety scores to encourage
    safer healthcare decision-making, especially important in critical
    care scenarios.

    Args:
        q_values (torch.Tensor): Current Q-values of shape [batch_size, num_actions].
        safety_scores (torch.Tensor): Safety scores for each action of shape [batch_size, num_actions].
        safety_threshold (float): Minimum safety score threshold. Defaults to 0.5.
        safety_weight (float): Weight for the safety regularization term. Defaults to 1.0.

    Returns:
        torch.Tensor: Safety regularization loss.
    """
    # Identify unsafe actions (below threshold)
    unsafe_mask = safety_scores < safety_threshold

    # Penalty for high Q-values on unsafe actions
    unsafe_q_values = q_values * unsafe_mask.float()
    safety_penalty = torch.sum(torch.relu(unsafe_q_values), dim=1).mean()

    return safety_weight * safety_penalty


@typechecked
def compute_clinical_consistency_loss(
    q_values: torch.Tensor,
    previous_q_values: torch.Tensor,
    consistency_weight: float = 0.1,
) -> torch.Tensor:
    """Compute clinical consistency regularization.

    Encourages temporal consistency in Q-value estimates to prevent
    erratic decision-making in healthcare scenarios where stability
    is often preferred over rapid policy changes.

    Args:
        q_values (torch.Tensor): Current Q-values of shape [batch_size, num_actions].
        previous_q_values (torch.Tensor): Q-values from previous time step of shape [batch_size, num_actions].
        consistency_weight (float): Weight for the consistency regularization. Defaults to 0.1.

    Returns:
        torch.Tensor: Clinical consistency loss.
    """
    # L2 penalty for large changes in Q-values
    consistency_loss = F.mse_loss(q_values, previous_q_values.detach())

    return consistency_weight * consistency_loss


class HealthcareCQLRegularizer:
    """Healthcare-specific CQL regularizer combining clinical constraints.

    This class combines standard CQL regularization with healthcare-specific
    penalties to encourage safer and more consistent clinical decision-making.

    Args:
        cql_alpha (float): CQL regularization coefficient.
        safety_weight (float): Weight for safety regularization.
        consistency_weight (float): Weight for temporal consistency.
        penalty_weight (float): Weight for invalid action penalty.
    """

    def __init__(
        self,
        cql_alpha: float = CQL_CONFIG["cql_alpha"],
        safety_weight: float = 0.5,
        consistency_weight: float = 0.1,
        penalty_weight: float = 1.0,
    ):
        self.cql_alpha = cql_alpha
        self.safety_weight = safety_weight
        self.consistency_weight = consistency_weight
        self.penalty_weight = penalty_weight
        self.previous_q_values = None

    def compute_total_loss(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        next_q_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        valid_actions_mask: torch.Tensor = None,
        safety_scores: torch.Tensor = None,
        gamma: float = 0.99,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total CQL loss with healthcare-specific regularization.

        This method computes the combined CQL loss along with additional
        healthcare penalties based on valid actions and safety scores, and
        enforces temporal consistency.

        Args:
            q_values (torch.Tensor): Current Q-values [batch_size, num_actions].
            actions (torch.Tensor): Actions taken [batch_size].
            next_q_values (torch.Tensor): Next state Q-values [batch_size, num_actions].
            rewards (torch.Tensor): Rewards received [batch_size].
            dones (torch.Tensor): Episode termination flags [batch_size].
            valid_actions_mask (Optional[torch.Tensor]): Mask for valid actions [batch_size, num_actions].
            safety_scores (Optional[torch.Tensor]): Safety scores for actions [batch_size, num_actions].
            gamma (float): Discount factor. Defaults to 0.99.

        Returns:
            total_loss (torch.Tensor): Combined loss including all penalties.
            loss_components (dict): Dictionary of individual loss components.
        """
        # Standard CQL loss
        total_loss, bellman_loss, cql_loss = compute_cql_loss(
            q_values, actions, next_q_values, rewards, dones, gamma, self.cql_alpha
        )

        loss_components = {"bellman_loss": bellman_loss, "cql_loss": cql_loss}

        # Healthcare-specific penalties
        if valid_actions_mask is not None:
            action_penalty = compute_healthcare_action_penalty(
                q_values, valid_actions_mask, self.penalty_weight
            )
            total_loss += action_penalty
            loss_components["action_penalty"] = action_penalty

        if safety_scores is not None:
            safety_penalty = compute_safety_regularization(
                q_values, safety_scores, safety_weight=self.safety_weight
            )
            total_loss += safety_penalty
            loss_components["safety_penalty"] = safety_penalty

        if self.previous_q_values is not None:
            consistency_loss = compute_clinical_consistency_loss(
                q_values, self.previous_q_values, self.consistency_weight
            )
            total_loss += consistency_loss
            loss_components["consistency_loss"] = consistency_loss

        # Update previous Q-values for next iteration
        self.previous_q_values = q_values.detach().clone()

        return total_loss, loss_components


@typechecked
def compute_adaptive_cql_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    next_q_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    states: torch.Tensor,
    gamma: float = 0.99,
    high_risk_alpha: float = 0.5,
    normal_alpha: float = 1.0,
    risk_threshold: float = 0.7,
    cql_temperature: float = CQL_CONFIG["cql_temperature"],
    num_random_actions: int = CQL_CONFIG["num_random_actions"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute adaptive CQL loss with risk-aware alpha values.

    Uses α = 0.5 for high-risk states (risk_score ≥ 0.7) and α = 1.0 otherwise.

    Args:
        q_values (torch.Tensor): Current Q-values of shape [batch_size, num_actions].
        actions (torch.Tensor): Actions taken of shape [batch_size].
        next_q_values (torch.Tensor): Next state Q-values of shape [batch_size, num_actions].
        rewards (torch.Tensor): Rewards received of shape [batch_size].
        dones (torch.Tensor): Episode termination flags of shape [batch_size].
        states (torch.Tensor): Patient states of shape [batch_size, state_dim].
        gamma (float): Discount factor. Defaults to 0.99.
        high_risk_alpha (float): CQL alpha for high-risk states. Defaults to 0.5.
        normal_alpha (float): CQL alpha for normal states. Defaults to 1.0.
        risk_threshold (float): Risk threshold for adaptive alpha. Defaults to 0.7.
        cql_temperature (float): Temperature for CQL logsumexp computation.
        num_random_actions (int): Number of random actions for CQL penalty.

    Returns:
        total_loss (torch.Tensor): Combined Bellman and adaptive CQL loss.
        bellman_loss (torch.Tensor): Bellman error loss.
        cql_loss (torch.Tensor): Adaptive CQL regularization loss.
    """
    # Handle different Q-value tensor shapes
    if q_values.dim() == 2:
        batch_size, num_actions = q_values.shape
    elif q_values.dim() == 3:
        # Distributional case: [batch_size, num_actions, num_atoms]
        batch_size, num_actions, _ = q_values.shape
        # Convert to expected Q-values by taking mean over atoms
        q_values = q_values.mean(dim=-1)
    else:
        raise ValueError(f"Unexpected Q-values shape: {q_values.shape}")

    device = q_values.device

    # Handle next_q_values shape as well
    if next_q_values.dim() == 3:
        # Distributional case: convert to Q-values
        next_q_values = next_q_values.mean(dim=-1)

    # Standard Bellman error (TD loss)
    q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_max = next_q_values.max(dim=1)[0]
        target_q_values = rewards + gamma * next_q_max * (~dones)

    bellman_loss = F.mse_loss(q_values_selected, target_q_values)

    # Extract risk scores from states (assuming risk_score is the second feature)
    risk_scores = states[:, 1] if states.shape[1] > 1 else torch.ones(batch_size, device=device) * 0.5

    # Determine adaptive alpha values
    high_risk_mask = risk_scores >= risk_threshold
    adaptive_alphas = torch.where(high_risk_mask, high_risk_alpha, normal_alpha)

    # CQL regularization term with adaptive alpha
    current_state_logsumexp = torch.logsumexp(q_values / cql_temperature, dim=1)

    # Generate random actions for out-of-distribution penalty
    random_actions = torch.randint(
        0, num_actions, (batch_size, num_random_actions), device=device
    )

    # Compute Q-values for random actions
    random_q_values = q_values.unsqueeze(1).expand(-1, num_random_actions, -1)
    random_q_selected = random_q_values.gather(2, random_actions.unsqueeze(2)).squeeze(2)
    random_logsumexp = torch.logsumexp(random_q_selected / cql_temperature, dim=1)

    # CQL penalty with adaptive alpha
    cql_penalty = current_state_logsumexp + random_logsumexp - 2 * q_values_selected
    cql_loss = (adaptive_alphas * cql_penalty).mean()

    # Total loss
    total_loss = bellman_loss + cql_loss

    return total_loss, bellman_loss, cql_loss


@typechecked
def compute_monotonicity_regularizer(
    q_values: torch.Tensor,
    states: torch.Tensor,
    refer_action_idx: int = 1,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute monotonicity regularizer for Q-values based on risk scores.

    Penalizes Q_refer(s_i) < Q_refer(s_j) whenever risk_i < risk_j to enforce
    that higher risk patients should have higher Q-values for referral actions.

    Args:
        q_values (torch.Tensor): Q-values of shape [batch_size, num_actions].
        states (torch.Tensor): Patient states of shape [batch_size, state_dim].
        refer_action_idx (int): Index of the REFER action. Defaults to 1.
        beta (float): Weight for monotonicity regularizer. Defaults to 0.1.

    Returns:
        torch.Tensor: Monotonicity regularization loss.
    """
    # Handle different Q-value tensor shapes
    if q_values.dim() == 2:
        batch_size = q_values.shape[0]
    elif q_values.dim() == 3:
        # Distributional case: [batch_size, num_actions, num_atoms]
        batch_size = q_values.shape[0]
        # Convert to expected Q-values by taking mean over atoms
        q_values = q_values.mean(dim=-1)
    else:
        raise ValueError(f"Unexpected Q-values shape: {q_values.shape}")

    if batch_size < 2:
        return torch.tensor(0.0, device=q_values.device)

    # Extract risk scores (assuming risk_score is the second feature)
    risk_scores = states[:, 1] if states.shape[1] > 1 else torch.ones(batch_size, device=states.device) * 0.5

    # Extract Q-values for REFER action
    q_refer = q_values[:, refer_action_idx]

    # Compute pairwise differences
    # risk_diff[i,j] = risk_i - risk_j
    risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)

    # q_diff[i,j] = Q_refer(s_i) - Q_refer(s_j)
    q_diff = q_refer.unsqueeze(1) - q_refer.unsqueeze(0)

    # Monotonicity violation: risk_i < risk_j but Q_refer(s_i) >= Q_refer(s_j)
    # This means risk_diff < 0 but q_diff >= 0
    violation_mask = (risk_diff < 0) & (q_diff >= 0)

    # Penalty for violations: penalize positive q_diff when risk_diff is negative
    violations = torch.relu(q_diff) * violation_mask.float()

    # Average over all pairs
    monotonicity_loss = violations.sum() / (batch_size * (batch_size - 1))

    return beta * monotonicity_loss
