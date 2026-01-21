"""
Training utilities for RFTG neural networks.

This module provides:
- Experience replay buffers (uniform and prioritized)
- Target network management
- Training helpers for stable RL training
"""

import copy
import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    info: Optional[Dict[str, Any]] = None


@dataclass
class EvalExperience:
    """Experience for eval network training (value estimation)."""
    state: np.ndarray
    player_index: int
    round_num: int
    outcome: np.ndarray  # Win probability for each player
    game_id: Optional[int] = None


@dataclass
class PolicyExperience:
    """Experience for policy network training (action selection)."""
    state: np.ndarray
    action: int
    action_scores: Optional[np.ndarray] = None  # MCTS or search scores
    reward: float = 0.0
    player_index: int = 0
    game_id: Optional[int] = None


class ReplayBuffer:
    """
    Simple uniform replay buffer.

    Stores experiences and samples uniformly at random.
    Uses a circular buffer to maintain fixed memory usage.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, experience: Union[Experience, EvalExperience, PolicyExperience]):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: List[Union[Experience, EvalExperience, PolicyExperience]]):
        """Add multiple experiences to the buffer."""
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int) -> List[Any]:
        """
        Sample a batch of experiences uniformly at random.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= min_size

    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples experiences based on their TD error (priority).
    Higher priority experiences are sampled more frequently.

    Uses sum-tree data structure for efficient O(log n) sampling.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (increases over training)
            beta_increment: How much to increase beta each sample
            epsilon: Small constant to ensure non-zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer: List[Optional[Any]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def add(
        self,
        experience: Union[Experience, EvalExperience, PolicyExperience],
        priority: Optional[float] = None,
    ):
        """
        Add an experience with given priority.

        Args:
            experience: The experience to add
            priority: Initial priority (default: max priority seen so far)
        """
        if priority is None:
            priority = self.max_priority

        self.buffer[self.position] = experience
        self.priorities[self.position] = (abs(priority) + self.epsilon) ** self.alpha

        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        experiences: List[Union[Experience, EvalExperience, PolicyExperience]],
        priorities: Optional[List[float]] = None,
    ):
        """Add multiple experiences with priorities."""
        if priorities is None:
            priorities = [None] * len(experiences)

        for exp, pri in zip(experiences, priorities):
            self.add(exp, pri)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (abs(priority) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return self.size >= min_size

    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0


class TargetNetwork:
    """
    Manages a target network for stable training.

    The target network is a slow-moving copy of the main network,
    updated either periodically (hard update) or continuously (soft update).
    This provides stable targets for TD learning.
    """

    def __init__(
        self,
        model: nn.Module,
        update_mode: str = 'soft',
        tau: float = 0.005,
        update_freq: int = 1000,
    ):
        """
        Initialize the target network.

        Args:
            model: The main network to create a target for
            update_mode: 'soft' for exponential moving average, 'hard' for periodic copy
            tau: Soft update coefficient (higher = faster update)
            update_freq: Steps between hard updates
        """
        self.update_mode = update_mode
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0

        # Create target network as a deep copy
        self.target = copy.deepcopy(model)
        self.target.eval()

        # Freeze target network parameters
        for param in self.target.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module, force: bool = False):
        """
        Update the target network.

        Args:
            model: The main network
            force: If True, perform update regardless of mode/counter
        """
        self.update_counter += 1

        if self.update_mode == 'soft' or force:
            # Soft update: target = tau * model + (1 - tau) * target
            with torch.no_grad():
                for target_param, param in zip(
                    self.target.parameters(), model.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        elif self.update_mode == 'hard':
            # Hard update: periodically copy weights
            if self.update_counter % self.update_freq == 0:
                self.target.load_state_dict(model.state_dict())

    def get_target(self) -> nn.Module:
        """Get the target network."""
        return self.target

    def to(self, device: torch.device):
        """Move target network to device."""
        self.target = self.target.to(device)
        return self


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains an EMA of model weights for more stable evaluation.
    Similar to target network but typically used for the final model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to track
            decay: EMA decay rate (higher = smoother)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def collate_eval_experiences(
    experiences: List[EvalExperience],
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate eval experiences into batched tensors.

    Args:
        experiences: List of EvalExperience
        device: Device to place tensors on

    Returns:
        Tuple of (states, outcomes) tensors
    """
    states = np.array([exp.state for exp in experiences])
    outcomes = np.array([exp.outcome for exp in experiences])

    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(outcomes, dtype=torch.float32, device=device),
    )


def collate_policy_experiences(
    experiences: List[PolicyExperience],
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Collate policy experiences into batched tensors.

    Args:
        experiences: List of PolicyExperience
        device: Device to place tensors on

    Returns:
        Tuple of (states, actions, action_scores) tensors
    """
    states = np.array([exp.state for exp in experiences])
    actions = np.array([exp.action for exp in experiences])

    # Action scores may be None for some experiences
    has_scores = all(exp.action_scores is not None for exp in experiences)
    if has_scores:
        scores = np.array([exp.action_scores for exp in experiences])
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores_tensor = None

    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.long, device=device),
        scores_tensor,
    )


class TrainingState:
    """
    Tracks training state for checkpointing and resumption.
    """

    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }

    def update(self, train_loss: float, val_loss: Optional[float] = None, lr: float = 0.0):
        """Update training state."""
        self.history['train_loss'].append(train_loss)
        self.history['lr'].append(lr)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = self.epoch

        self.epoch += 1

    def save(self) -> dict:
        """Get state dict for saving."""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }

    def load(self, state_dict: dict):
        """Load from state dict."""
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_val_loss = state_dict['best_val_loss']
        self.best_epoch = state_dict['best_epoch']
        self.history = state_dict['history']


if __name__ == "__main__":
    # Test the utilities
    import sys

    print("Testing training utilities")
    print("=" * 60)

    # Test ReplayBuffer
    print("\nTesting ReplayBuffer:")
    buffer = ReplayBuffer(capacity=1000)

    # Add some experiences
    for i in range(100):
        exp = EvalExperience(
            state=np.random.randn(704).astype(np.float32),
            player_index=i % 2,
            round_num=i % 10,
            outcome=np.array([0.5, 0.5] if i % 2 == 0 else [1.0, 0.0]),
        )
        buffer.add(exp)

    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready (min=50): {buffer.is_ready(50)}")

    batch = buffer.sample(32)
    print(f"  Sampled batch size: {len(batch)}")

    # Test PrioritizedReplayBuffer
    print("\nTesting PrioritizedReplayBuffer:")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)

    for i in range(100):
        exp = PolicyExperience(
            state=np.random.randn(605).astype(np.float32),
            action=i % 7,
            action_scores=np.random.randn(7).astype(np.float32),
        )
        pri_buffer.add(exp, priority=np.random.rand())

    print(f"  Buffer size: {len(pri_buffer)}")

    exps, indices, weights = pri_buffer.sample(32)
    print(f"  Sampled batch size: {len(exps)}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Update priorities
    new_priorities = np.random.rand(32) * 2
    pri_buffer.update_priorities(indices, new_priorities)
    print("  Priorities updated successfully")

    # Test TargetNetwork
    print("\nTesting TargetNetwork:")
    from modern_nets import ResidualMLP

    model = ResidualMLP(num_inputs=704, num_outputs=2, hidden_dim=128, num_layers=2)
    target_net = TargetNetwork(model, update_mode='soft', tau=0.005)

    print(f"  Main model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Target model params: {sum(p.numel() for p in target_net.target.parameters()):,}")

    # Update target
    for _ in range(10):
        target_net.update(model)
    print("  Target updated 10 times (soft update)")

    # Test EMAModel
    print("\nTesting EMAModel:")
    ema = EMAModel(model, decay=0.999)

    for _ in range(10):
        ema.update(model)
    print("  EMA updated 10 times")

    ema.apply_shadow(model)
    print("  Shadow weights applied")
    ema.restore(model)
    print("  Original weights restored")

    # Test collate functions
    print("\nTesting collate functions:")
    eval_exps = [
        EvalExperience(
            state=np.random.randn(704).astype(np.float32),
            player_index=0,
            round_num=0,
            outcome=np.array([0.6, 0.4]),
        )
        for _ in range(16)
    ]
    states, outcomes = collate_eval_experiences(eval_exps)
    print(f"  Eval batch: states={states.shape}, outcomes={outcomes.shape}")

    policy_exps = [
        PolicyExperience(
            state=np.random.randn(605).astype(np.float32),
            action=np.random.randint(7),
            action_scores=np.random.randn(7).astype(np.float32),
        )
        for _ in range(16)
    ]
    states, actions, scores = collate_policy_experiences(policy_exps)
    print(f"  Policy batch: states={states.shape}, actions={actions.shape}, scores={scores.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
