#!/usr/bin/env python3
"""
PyTorch training script for RFTG neural networks.

Trains eval and role networks from self-play data exported by the C learner.

Supports multiple architectures:
- baseline: Original 2-layer MLP (704 → 50 → 2)
- residual-small: ResidualMLP with 128 hidden, 2 layers (~200K params)
- residual-medium: ResidualMLP with 256 hidden, 4 layers (~1.1M params)
- residual-large: ResidualMLP with 512 hidden, 6 layers (~4.5M params)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

from rftg_net import RFTGNet, load_net_file
from training_data import (
    BinaryBatchReader, EVAL_INPUT_DIM, ROLE_INPUT_DIM, ROLE_BINARY_DIM, ROLE_FLOAT_DIM
)
from modern_nets import get_architecture, print_model_summary, ResidualMLP, UnifiedRFTGNet
from training_utils import TargetNetwork, EMAModel

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class RFTGEvalDataset(Dataset):
    """Dataset for training the eval (win probability) network."""

    def __init__(self, data_dirs: Union[str, List[str]], max_games: Optional[int] = None):
        """
        Load eval training data from JSONL or NPZ files.

        Automatically detects format based on file extensions:
        - .npz files: Binary format (faster, smaller)
        - .jsonl files: JSON format (legacy)

        Args:
            data_dirs: Directory or list of directories containing training files
            max_games: Maximum number of games to load (None = all)
        """
        self.inputs = []
        self.targets = []

        # Handle single dir or list of dirs
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        games_loaded = 0

        for data_dir in data_dirs:
            data_path = Path(data_dir)

            # Check for NPZ files first (preferred format)
            npz_files = sorted(data_path.glob("*.npz"))
            if npz_files:
                games_loaded = self._load_from_npz(npz_files, max_games, games_loaded)
            else:
                # Fall back to JSONL format
                jsonl_files = sorted(data_path.glob("*.jsonl"))
                games_loaded = self._load_from_jsonl(jsonl_files, max_games, games_loaded)

            if max_games and games_loaded >= max_games:
                break

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        print(f"Loaded {len(self.inputs)} eval states from {games_loaded} games")

    def _load_from_npz(self, npz_files: List[Path], max_games: Optional[int], games_loaded: int) -> int:
        """Load eval data from NPZ files."""
        for filepath in npz_files:
            if max_games and games_loaded >= max_games:
                break

            with BinaryBatchReader(str(filepath)) as reader:
                # Get game info for targets
                game_info = reader.get_game_info()
                boundaries = reader.get_game_boundaries()
                eval_inputs, eval_meta = reader.get_eval_data()

                for i, info in enumerate(game_info):
                    if max_games and games_loaded >= max_games:
                        break

                    # Get sample boundaries for this game
                    eval_start = boundaries[i, 0]
                    eval_end = boundaries[i + 1, 0] if i + 1 < len(boundaries) else len(eval_inputs)

                    # Compute targets from game outcome
                    num_players = info['num_players']
                    winners = set(info['winner_indices'])
                    target = [1.0 / len(winners) if j in winners else 0.0
                              for j in range(num_players)]

                    # Add samples for this game
                    for j in range(eval_start, eval_end):
                        self.inputs.append(eval_inputs[j].tolist())
                        player_idx = int(eval_meta[j, 0])
                        rotated_target = target[player_idx:] + target[:player_idx]
                        self.targets.append(rotated_target)

                    games_loaded += 1

        return games_loaded

    def _load_from_jsonl(self, jsonl_files: List[Path], max_games: Optional[int], games_loaded: int) -> int:
        """Load eval data from JSONL files (legacy format)."""
        for filepath in jsonl_files:
            with open(filepath, 'r') as f:
                for line in f:
                    if max_games and games_loaded >= max_games:
                        break
                    try:
                        game = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Get winner info
                    num_players = game['num_players']
                    winners = set(game['winner_indices'])

                    # Create target: 1.0 for winners, 0.0 for losers
                    target = [1.0 / len(winners) if i in winners else 0.0
                              for i in range(num_players)]

                    # Add each eval state with the game outcome as target
                    for state in game['eval_states']:
                        self.inputs.append(state['inputs'])
                        player_idx = state['player_index']
                        rotated_target = target[player_idx:] + target[:player_idx]
                        self.targets.append(rotated_target)

                    games_loaded += 1

            if max_games and games_loaded >= max_games:
                break

        return games_loaded

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class RFTGRoleDataset(Dataset):
    """Dataset for training the role (action selection) network."""

    def __init__(self, data_dirs: Union[str, List[str]], max_games: Optional[int] = None):
        """
        Load role training data from JSONL or NPZ files.

        Automatically detects format based on file extensions:
        - .npz files: Binary format (faster, smaller)
        - .jsonl files: JSON format (legacy)

        Args:
            data_dirs: Directory or list of directories containing training files
            max_games: Maximum number of games to load (None = all)
        """
        self.inputs = []
        self.chosen_actions = []
        self.num_actions_list = []

        # Handle single dir or list of dirs
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        games_loaded = 0

        for data_dir in data_dirs:
            data_path = Path(data_dir)

            # Check for NPZ files first (preferred format)
            npz_files = sorted(data_path.glob("*.npz"))
            if npz_files:
                games_loaded = self._load_from_npz(npz_files, max_games, games_loaded)
            else:
                # Fall back to JSONL format
                jsonl_files = sorted(data_path.glob("*.jsonl"))
                games_loaded = self._load_from_jsonl(jsonl_files, max_games, games_loaded)

            if max_games and games_loaded >= max_games:
                break

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.chosen_actions = torch.tensor(self.chosen_actions, dtype=torch.long)

        print(f"Loaded {len(self.inputs)} role decisions from {games_loaded} games")

    def _load_from_npz(self, npz_files: List[Path], max_games: Optional[int], games_loaded: int) -> int:
        """Load role data from NPZ files."""
        for filepath in npz_files:
            if max_games and games_loaded >= max_games:
                break

            with BinaryBatchReader(str(filepath)) as reader:
                # Get game boundaries and data
                boundaries = reader.get_game_boundaries()
                role_inputs = reader.get_full_role_inputs()  # Binary + floats concatenated
                _, role_floats, role_meta = reader.get_role_data()

                num_games_in_file = len(boundaries)
                for i in range(num_games_in_file):
                    if max_games and games_loaded >= max_games:
                        break

                    # Get sample boundaries for this game
                    role_start = boundaries[i, 1]
                    role_end = boundaries[i + 1, 1] if i + 1 < len(boundaries) else len(role_inputs)

                    # Add samples for this game
                    for j in range(role_start, role_end):
                        self.inputs.append(role_inputs[j].tolist())
                        self.chosen_actions.append(int(role_meta[j, 2]))
                        self.num_actions_list.append(len(role_floats[j]))

                    games_loaded += 1

        return games_loaded

    def _load_from_jsonl(self, jsonl_files: List[Path], max_games: Optional[int], games_loaded: int) -> int:
        """Load role data from JSONL files (legacy format)."""
        for filepath in jsonl_files:
            with open(filepath, 'r') as f:
                for line in f:
                    if max_games and games_loaded >= max_games:
                        break
                    try:
                        game = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    for decision in game['role_decisions']:
                        self.inputs.append(decision['inputs'])
                        self.chosen_actions.append(decision['chosen_action'])
                        self.num_actions_list.append(len(decision['action_scores']))

                    games_loaded += 1

            if max_games and games_loaded >= max_games:
                break

        return games_loaded

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.chosen_actions[idx]


def compute_eval_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute validation loss for eval network."""
    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.kl_div(
                torch.log(outputs + 1e-10),
                targets,
                reduction='batchmean'
            )
            val_loss += loss.item() * inputs.size(0)
            val_samples += inputs.size(0)

    return val_loss / val_samples if val_samples > 0 else float('inf')


def train_eval_network(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-5,
    scheduler_type: str = 'cosine',
    early_stopping_patience: int = 5,
    checkpoint_dir: Optional[str] = None,
    model_name: str = 'eval',
    max_examples: Optional[int] = None,
    use_wandb: bool = False,
) -> Dict[str, List[float]]:
    """
    Train the eval network with modern training techniques.

    Uses KL divergence loss to match predicted win probabilities
    to actual game outcomes.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        device: Device to train on
        weight_decay: L2 regularization weight
        scheduler_type: LR scheduler type ('cosine', 'plateau', or 'none')
        early_stopping_patience: Epochs to wait before early stopping (0 to disable)
        checkpoint_dir: Directory to save checkpoints (None to disable)
        model_name: Name prefix for saved models
        max_examples: Stop after training on this many examples (None = no limit)
        use_wandb: Whether to log metrics to Weights & Biases

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = None

    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Compute and print initial validation loss
    if val_loader:
        init_val_loss = compute_eval_val_loss(model, val_loader, device)
        print(f"Initial validation loss: {init_val_loss:.6f}")
        history['val_loss'].append(init_val_loss)
        if use_wandb and wandb:
            wandb.log({"val_loss": init_val_loss, "epoch": 0, "examples": 0})

    # Early stopping tracking
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    total_examples_trained = 0
    stopped_by_max_examples = False

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # KL divergence loss (targets are the "true" distribution)
            # Add small epsilon for numerical stability
            loss = F.kl_div(
                torch.log(outputs + 1e-10),
                targets,
                reduction='batchmean'
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            total_examples_trained += inputs.size(0)

            # Check if we've hit max_examples
            if max_examples and total_examples_trained >= max_examples:
                stopped_by_max_examples = True
                break

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation
        if val_loader:
            avg_val_loss = compute_eval_val_loss(model, val_loader, device)
            history['val_loss'].append(avg_val_loss)

            # Update scheduler (for plateau scheduler, need val loss)
            if scheduler_type == 'plateau' and scheduler:
                scheduler.step(avg_val_loss)
            elif scheduler_type == 'cosine' and scheduler:
                scheduler.step()

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'train_loss': avg_train_loss,
                    }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            # Log progress
            lr_str = f", lr={optimizer.param_groups[0]['lr']:.2e}"
            examples_str = f" [{total_examples_trained} examples]" if max_examples else ""
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={avg_val_loss:.6f}{lr_str}{examples_str}"
                  f"{' *' if epoch == best_epoch else ''}")

            # Wandb logging
            if use_wandb and wandb:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "examples": total_examples_trained,
                    "best_val_loss": best_val_loss,
                })

            # Early stopping by patience
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
                break
        else:
            # No validation, just update cosine scheduler
            if scheduler_type == 'cosine' and scheduler:
                scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if use_wandb and wandb:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "examples": total_examples_trained,
                })

        # Stop if we hit max_examples
        if stopped_by_max_examples:
            print(f"Stopped after {total_examples_trained} training examples")
            if val_loader:
                final_val_loss = compute_eval_val_loss(model, val_loader, device)
                print(f"Final validation loss: {final_val_loss:.6f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch+1} (val_loss={best_val_loss:.6f})")

    return history


def overfit_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_steps: int,
    lr: float,
    device: torch.device,
    print_every: int = 100,
) -> Dict[str, List[float]]:
    """
    Overfit on a batch of examples as a sanity check.

    This tests whether the network has enough capacity to memorize
    a small batch, which is a basic requirement for any learning.

    Args:
        model: Neural network model
        inputs: Input tensor of shape (batch_size, num_inputs)
        targets: Target tensor of shape (batch_size, num_outputs)
        num_steps: Number of gradient steps
        lr: Learning rate
        device: Device to train on
        print_every: Print loss every N steps

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    inputs = inputs.to(device)
    targets = targets.to(device)
    batch_size = inputs.shape[0]

    history = {'loss': []}

    print(f"Overfitting on batch of {batch_size} examples for {num_steps} steps...")
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Initial loss
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        init_loss = F.kl_div(
            torch.log(output + 1e-10),
            targets,
            reduction='batchmean'
        ).item()
    print(f"Step 0: loss={init_loss:.6f}")
    history['loss'].append(init_loss)

    # Training loop
    model.train()
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        output = model(inputs)

        loss = F.kl_div(
            torch.log(output + 1e-10),
            targets,
            reduction='batchmean'
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())

        if step % print_every == 0 or step == num_steps:
            print(f"Step {step}: loss={loss.item():.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_output = model(inputs)
        final_loss = F.kl_div(
            torch.log(final_output + 1e-10),
            targets,
            reduction='batchmean'
        ).item()

        # Check per-example accuracy
        pred_winners = final_output.argmax(dim=1)
        true_winners = targets.argmax(dim=1)
        accuracy = (pred_winners == true_winners).float().mean().item()

    print(f"\nFinal: loss={final_loss:.6f}, accuracy={accuracy:.2%}")

    # Show a few examples
    print("\nSample predictions (first 5):")
    for i in range(min(5, batch_size)):
        print(f"  Target: {targets[i].tolist()} -> Output: {[f'{x:.3f}' for x in final_output[i].tolist()]}")

    if final_loss < 0.01:
        print("\nSUCCESS: Model can overfit batch!")
    elif final_loss < 0.1:
        print("\nPARTIAL: Model partially overfit batch")
    else:
        print("\nWARNING: Model failed to overfit batch")

    return history


def train_role_network(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-5,
    scheduler_type: str = 'cosine',
    early_stopping_patience: int = 5,
    checkpoint_dir: Optional[str] = None,
    model_name: str = 'role',
) -> Dict[str, List[float]]:
    """
    Train the role network with modern training techniques.

    Uses cross-entropy loss to predict the chosen action.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        device: Device to train on
        weight_decay: L2 regularization weight
        scheduler_type: LR scheduler type ('cosine', 'plateau', or 'none')
        early_stopping_patience: Epochs to wait before early stopping (0 to disable)
        checkpoint_dir: Directory to save checkpoints (None to disable)
        model_name: Name prefix for saved models

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Set up learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # Early stopping tracking
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_samples += inputs.size(0)

        avg_train_loss = train_loss / train_samples
        train_acc = train_correct / train_samples
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                    val_samples += inputs.size(0)

            avg_val_loss = val_loss / val_samples
            val_acc = val_correct / val_samples
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            # Update scheduler
            if scheduler_type == 'plateau' and scheduler:
                scheduler.step(avg_val_loss)
            elif scheduler_type == 'cosine' and scheduler:
                scheduler.step()

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'train_loss': avg_train_loss,
                    }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}"
                  f"{' *' if epoch == best_epoch else ''}")

            # Early stopping
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
                break
        else:
            if scheduler_type == 'cosine' and scheduler:
                scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch+1} (val_loss={best_val_loss:.4f})")

    return history


class RFTGUnifiedDataset(Dataset):
    """
    Dataset for training unified (value+policy) network.

    Each sample contains both eval state (for value head) and
    the corresponding action decision (for policy head).

    Since eval and policy inputs have different dimensions, we pad
    the smaller inputs to match the larger dimension (filled with zeros).
    """

    def __init__(self, data_dirs: Union[str, List[str]], max_games: Optional[int] = None):
        """
        Load unified training data from JSONL or NPZ files.

        Automatically detects format based on file extensions:
        - .npz files: Binary format (faster, smaller)
        - .jsonl files: JSON format (legacy)

        Args:
            data_dirs: Directory or list of directories containing training files
            max_games: Maximum number of games to load (None = all)
        """
        eval_inputs_raw = []
        self.eval_targets = []
        policy_inputs_raw = []
        self.policy_targets = []

        # Handle single dir or list of dirs
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        games_loaded = 0

        for data_dir in data_dirs:
            data_path = Path(data_dir)

            # Check for NPZ files first (preferred format)
            npz_files = sorted(data_path.glob("*.npz"))
            if npz_files:
                games_loaded = self._load_from_npz(
                    npz_files, max_games, games_loaded,
                    eval_inputs_raw, policy_inputs_raw
                )
            else:
                # Fall back to JSONL format
                jsonl_files = sorted(data_path.glob("*.jsonl"))
                games_loaded = self._load_from_jsonl(
                    jsonl_files, max_games, games_loaded,
                    eval_inputs_raw, policy_inputs_raw
                )

            if max_games and games_loaded >= max_games:
                break

        # Get dimensions
        self.eval_input_dim = len(eval_inputs_raw[0]) if eval_inputs_raw else 0
        self.policy_input_dim = len(policy_inputs_raw[0]) if policy_inputs_raw else 0
        self.max_input_dim = max(self.eval_input_dim, self.policy_input_dim)

        # Pad inputs to max dimension
        def pad_inputs(inputs, target_dim):
            if len(inputs[0]) == target_dim:
                return inputs
            # Pad with zeros
            return [inp + [0.0] * (target_dim - len(inp)) for inp in inputs]

        eval_inputs_padded = pad_inputs(eval_inputs_raw, self.max_input_dim)
        policy_inputs_padded = pad_inputs(policy_inputs_raw, self.max_input_dim)

        self.eval_inputs = torch.tensor(eval_inputs_padded, dtype=torch.float32)
        self.eval_targets = torch.tensor(self.eval_targets, dtype=torch.float32)
        self.policy_inputs = torch.tensor(policy_inputs_padded, dtype=torch.float32)
        self.policy_targets = torch.tensor(self.policy_targets, dtype=torch.long)

        print(f"Loaded {len(self.eval_inputs)} eval states and "
              f"{len(self.policy_inputs)} policy decisions from {games_loaded} games")
        if self.eval_input_dim != self.policy_input_dim:
            print(f"  (padded inputs from {self.eval_input_dim}/{self.policy_input_dim} to {self.max_input_dim})")

        # Store dimensions
        self.num_inputs = self.max_input_dim
        self.num_value_outputs = self.eval_targets.shape[1]
        self.num_policy_outputs = self.policy_targets.max().item() + 1

    def _load_from_npz(self, npz_files: List[Path], max_games: Optional[int], games_loaded: int,
                       eval_inputs_raw: list, policy_inputs_raw: list) -> int:
        """Load unified data from NPZ files."""
        for filepath in npz_files:
            if max_games and games_loaded >= max_games:
                break

            with BinaryBatchReader(str(filepath)) as reader:
                # Get all data
                game_info = reader.get_game_info()
                boundaries = reader.get_game_boundaries()
                eval_inputs, eval_meta = reader.get_eval_data()
                role_inputs = reader.get_full_role_inputs()
                _, _, role_meta = reader.get_role_data()

                for i, info in enumerate(game_info):
                    if max_games and games_loaded >= max_games:
                        break

                    # Get sample boundaries for this game
                    eval_start = boundaries[i, 0]
                    eval_end = boundaries[i + 1, 0] if i + 1 < len(boundaries) else len(eval_inputs)
                    role_start = boundaries[i, 1]
                    role_end = boundaries[i + 1, 1] if i + 1 < len(boundaries) else len(role_inputs)

                    # Compute targets from game outcome
                    num_players = info['num_players']
                    winners = set(info['winner_indices'])
                    target = [1.0 / len(winners) if j in winners else 0.0
                              for j in range(num_players)]

                    # Add eval samples
                    for j in range(eval_start, eval_end):
                        eval_inputs_raw.append(eval_inputs[j].tolist())
                        player_idx = int(eval_meta[j, 0])
                        rotated_target = target[player_idx:] + target[:player_idx]
                        self.eval_targets.append(rotated_target)

                    # Add policy samples
                    for j in range(role_start, role_end):
                        policy_inputs_raw.append(role_inputs[j].tolist())
                        self.policy_targets.append(int(role_meta[j, 2]))

                    games_loaded += 1

        return games_loaded

    def _load_from_jsonl(self, jsonl_files: List[Path], max_games: Optional[int], games_loaded: int,
                         eval_inputs_raw: list, policy_inputs_raw: list) -> int:
        """Load unified data from JSONL files (legacy format)."""
        for filepath in jsonl_files:
            with open(filepath, 'r') as f:
                for line in f:
                    if max_games and games_loaded >= max_games:
                        break
                    try:
                        game = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Get winner info for eval targets
                    num_players = game['num_players']
                    winners = set(game['winner_indices'])
                    target = [1.0 / len(winners) if i in winners else 0.0
                              for i in range(num_players)]

                    # Add eval states
                    for state in game['eval_states']:
                        eval_inputs_raw.append(state['inputs'])
                        player_idx = state['player_index']
                        rotated_target = target[player_idx:] + target[:player_idx]
                        self.eval_targets.append(rotated_target)

                    # Add policy decisions
                    for decision in game['role_decisions']:
                        policy_inputs_raw.append(decision['inputs'])
                        self.policy_targets.append(decision['chosen_action'])

                    games_loaded += 1

            if max_games and games_loaded >= max_games:
                break

        return games_loaded

    def __len__(self):
        # Return length of the smaller dataset
        return min(len(self.eval_inputs), len(self.policy_inputs))

    def __getitem__(self, idx):
        # For unified training, we need to handle different input sizes
        # Since eval and policy inputs have different dimensions,
        # we return both separately
        eval_idx = idx % len(self.eval_inputs)
        policy_idx = idx % len(self.policy_inputs)

        return {
            'eval_input': self.eval_inputs[eval_idx],
            'eval_target': self.eval_targets[eval_idx],
            'policy_input': self.policy_inputs[policy_idx],
            'policy_target': self.policy_targets[policy_idx],
        }


def train_unified_network(
    model: UnifiedRFTGNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-5,
    scheduler_type: str = 'cosine',
    early_stopping_patience: int = 5,
    checkpoint_dir: Optional[str] = None,
    model_name: str = 'unified',
    value_loss_weight: float = 1.0,
    policy_loss_weight: float = 1.0,
    use_ema: bool = True,
    ema_decay: float = 0.999,
) -> Dict[str, List[float]]:
    """
    Train a unified network with both value and policy heads.

    Uses KL divergence for value head and cross-entropy for policy head.
    Optionally uses EMA for more stable model weights.

    Args:
        model: UnifiedRFTGNet model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        device: Device to train on
        weight_decay: L2 regularization weight
        scheduler_type: LR scheduler type
        early_stopping_patience: Epochs without improvement before stopping
        checkpoint_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
        value_loss_weight: Weight for value loss
        policy_loss_weight: Weight for policy loss
        use_ema: Whether to use EMA for model weights
        ema_decay: EMA decay rate

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = None

    # EMA model
    ema = EMAModel(model, decay=ema_decay) if use_ema else None

    history = {
        'train_loss': [], 'train_value_loss': [], 'train_policy_loss': [],
        'val_loss': [], 'val_value_loss': [], 'val_policy_loss': [],
        'val_value_acc': [], 'val_policy_acc': [], 'lr': []
    }

    # Early stopping tracking
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_value_loss = 0.0
        train_policy_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            eval_inputs = batch['eval_input'].to(device)
            eval_targets = batch['eval_target'].to(device)
            policy_inputs = batch['policy_input'].to(device)
            policy_targets = batch['policy_target'].to(device)

            optimizer.zero_grad()

            # Forward pass for value - use only eval inputs
            value_output, _ = model(eval_inputs)
            value_loss = F.kl_div(
                torch.log(value_output + 1e-10),
                eval_targets,
                reduction='batchmean'
            )

            # Forward pass for policy - use only policy inputs
            _, policy_output = model(policy_inputs, return_logits=True)
            policy_loss = F.cross_entropy(policy_output, policy_targets)

            # Combined loss
            loss = value_loss_weight * value_loss + policy_loss_weight * policy_loss

            loss.backward()
            optimizer.step()

            if ema:
                ema.update(model)

            train_value_loss += value_loss.item() * eval_inputs.size(0)
            train_policy_loss += policy_loss.item() * policy_inputs.size(0)
            train_samples += eval_inputs.size(0)

        avg_train_value_loss = train_value_loss / train_samples
        avg_train_policy_loss = train_policy_loss / train_samples
        avg_train_loss = value_loss_weight * avg_train_value_loss + policy_loss_weight * avg_train_policy_loss

        history['train_loss'].append(avg_train_loss)
        history['train_value_loss'].append(avg_train_value_loss)
        history['train_policy_loss'].append(avg_train_policy_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation
        if val_loader:
            model.eval()

            # Apply EMA weights for validation
            if ema:
                ema.apply_shadow(model)

            val_value_loss = 0.0
            val_policy_loss = 0.0
            val_value_correct = 0
            val_policy_correct = 0
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    eval_inputs = batch['eval_input'].to(device)
                    eval_targets = batch['eval_target'].to(device)
                    policy_inputs = batch['policy_input'].to(device)
                    policy_targets = batch['policy_target'].to(device)

                    value_output, _ = model(eval_inputs)
                    _, policy_output = model(policy_inputs)

                    val_value_loss += F.kl_div(
                        torch.log(value_output + 1e-10),
                        eval_targets,
                        reduction='batchmean'
                    ).item() * eval_inputs.size(0)

                    val_policy_loss += F.cross_entropy(
                        policy_output, policy_targets
                    ).item() * policy_inputs.size(0)

                    val_value_correct += (value_output.argmax(dim=1) == eval_targets.argmax(dim=1)).sum().item()
                    val_policy_correct += (policy_output.argmax(dim=1) == policy_targets).sum().item()
                    val_samples += eval_inputs.size(0)

            # Restore original weights after validation
            if ema:
                ema.restore(model)

            avg_val_value_loss = val_value_loss / val_samples
            avg_val_policy_loss = val_policy_loss / val_samples
            avg_val_loss = value_loss_weight * avg_val_value_loss + policy_loss_weight * avg_val_policy_loss
            val_value_acc = val_value_correct / val_samples
            val_policy_acc = val_policy_correct / val_samples

            history['val_loss'].append(avg_val_loss)
            history['val_value_loss'].append(avg_val_value_loss)
            history['val_policy_loss'].append(avg_val_policy_loss)
            history['val_value_acc'].append(val_value_acc)
            history['val_policy_acc'].append(val_policy_acc)

            # Update scheduler
            if scheduler_type == 'plateau' and scheduler:
                scheduler.step(avg_val_loss)
            elif scheduler_type == 'cosine' and scheduler:
                scheduler.step()

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model state (with EMA weights if available)
                if ema:
                    ema.apply_shadow(model)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ema:
                    ema.restore(model)

                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'train_loss': avg_train_loss,
                    }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            # Log progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f} (v={avg_train_value_loss:.4f}, p={avg_train_policy_loss:.4f}), "
                  f"val_loss={avg_val_loss:.4f}, val_acc=(v={val_value_acc:.3f}, p={val_policy_acc:.3f})"
                  f"{' *' if epoch == best_epoch else ''}")

            # Early stopping
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
                break
        else:
            if scheduler_type == 'cosine' and scheduler:
                scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch+1} (val_loss={best_val_loss:.4f})")

    return history


def save_net_file(model: RFTGNet, filepath: str, input_names: Optional[List[str]] = None):
    """
    Save a PyTorch model to .net file format compatible with the C code.

    Args:
        model: Trained RFTGNet model
        filepath: Output path
        input_names: List of input names (optional, defaults to model's or numbered)
    """
    # Use model's input_names if available, otherwise generate default names
    if input_names is None:
        if hasattr(model, 'input_names') and model.input_names:
            input_names = model.input_names
        else:
            input_names = [f"input_{i}" for i in range(model.num_inputs)]

    with open(filepath, 'w') as f:
        # Header
        f.write(f"{model.num_inputs} {model.num_hidden} {model.num_outputs}\n")
        f.write(f"{getattr(model, 'num_training', 0)}\n")

        # Input names - MUST write exactly num_inputs names
        for i in range(model.num_inputs):
            if i < len(input_names):
                f.write(f"{input_names[i]}\n")
            else:
                f.write(f"input_{i}\n")

        # Hidden weights
        # PyTorch: hidden.weight is (num_hidden, num_inputs), hidden.bias is (num_hidden,)
        # C format: for each hidden node i, for each input+bias j, write weight[j][i]
        hidden_weight = model.hidden.weight.detach().cpu().numpy()  # (num_hidden, num_inputs)
        hidden_bias = model.hidden.bias.detach().cpu().numpy()  # (num_hidden,)

        for i in range(model.num_hidden):
            for j in range(model.num_inputs):
                f.write(f"{hidden_weight[i, j]}\n")
            f.write(f"{hidden_bias[i]}\n")

        # Output weights
        output_weight = model.output.weight.detach().cpu().numpy()  # (num_outputs, num_hidden)
        output_bias = model.output.bias.detach().cpu().numpy()  # (num_outputs,)

        for i in range(model.num_outputs):
            for j in range(model.num_hidden):
                f.write(f"{output_weight[i, j]}\n")
            f.write(f"{output_bias[i]}\n")

    print(f"Saved model to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RFTG neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline eval network
  python train.py -d /tmp/rftg_training_5min -n eval --arch baseline --epochs 30

  # Train medium residual network with checkpointing
  python train.py -d /tmp/rftg_training_5min -n eval --arch residual-medium --epochs 30

  # Train with specific hyperparameters
  python train.py -d ./data -n eval --arch residual-small --lr 0.0003 --weight-decay 1e-4

  # Train unified network (value+policy heads) with EMA
  python train.py -d ./data -n unified --arch unified-medium --epochs 30

  # Train unified with custom loss weights
  python train.py -d ./data -n unified --arch unified-large --value-loss-weight 1.0 --policy-loss-weight 0.5
        """
    )
    parser.add_argument("--data-dir", "-d", type=str, nargs='+', required=True,
                        help="Directory/directories containing training data (*.jsonl)")
    parser.add_argument("--output-dir", "-o", type=str, default="./trained_models",
                        help="Directory to save trained models")
    parser.add_argument("--network", "-n", choices=["eval", "role", "both", "unified"], default="both",
                        help="Which network to train (unified = combined value+policy heads)")
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="Maximum number of training epochs (default: 30)")
    parser.add_argument("--batch-size", "-b", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Maximum games to load (for testing)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Stop training after this many examples (for quick tests)")
    parser.add_argument("--overfit-batch", type=int, default=None, metavar="N",
                        help="Overfit on N examples (sanity check, default batch-size if 0)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--pretrained", "-p", type=str, default=None,
                        help="Path to pretrained .net file to continue training")
    parser.add_argument("--hidden", type=int, default=50,
                        help="Number of hidden nodes (for baseline architecture)")

    # New arguments for modern training
    parser.add_argument("--arch", "--architecture", type=str, default="baseline",
                        choices=["baseline", "residual-small", "residual-medium", "residual-large",
                                 "unified-small", "unified-medium", "unified-large"],
                        help="Network architecture to use (default: baseline)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (L2 regularization) coefficient")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Early stopping patience (epochs without improvement, 0 to disable)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable model checkpointing")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="rftg-training",
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name (auto-generated if not specified)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Wandb entity (username or team)")

    # Device selection
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "auto"],
                        help="Device to use for training (default: cuda)")

    # Unified network specific arguments
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use Exponential Moving Average for unified training")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate (default: 0.999)")
    parser.add_argument("--value-loss-weight", type=float, default=1.0,
                        help="Weight for value loss in unified training")
    parser.add_argument("--policy-loss-weight", type=float, default=1.0,
                        help="Weight for policy loss in unified training")

    args = parser.parse_args()

    checkpoint_dir = None if args.no_checkpoint else args.output_dir

    # Device selection - default to CUDA
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available!")
            print("Install PyTorch with CUDA support or use --device cpu")
            sys.exit(1)
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif args.device == "cpu":
        device = torch.device("cpu")
        print(f"Using device: {device}")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Install with: pip install wandb")
    if use_wandb:
        # Read API key from file if it exists
        wandb_key_path = Path.home() / ".wandb_rftg_key.txt"
        if wandb_key_path.exists():
            api_key = wandb_key_path.read_text().strip()
            wandb.login(key=api_key)

        run_name = args.wandb_run_name or f"{args.network}_{args.arch}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "architecture": args.arch,
                "network_type": args.network,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "early_stopping": args.early_stopping,
                "max_games": args.max_games,
                "max_examples": args.max_examples,
                "val_split": args.val_split,
            }
        )
        print(f"Wandb initialized: {wandb.run.name}")

    # Train eval network
    if args.network in ["eval", "both"]:
        print("\n" + "=" * 60)
        print("TRAINING EVAL NETWORK")
        print("=" * 60)

        # Load data
        print("Loading eval training data...")
        eval_dataset = RFTGEvalDataset(args.data_dir, max_games=args.max_games)

        # Split into train/val
        val_size = int(len(eval_dataset) * args.val_split)
        train_size = len(eval_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            eval_dataset, [train_size, val_size]
        )

        # Use pin_memory for faster CPU->GPU transfers when using CUDA
        pin_memory = device.type == "cuda"
        num_workers = 4 if device.type == "cuda" else 0
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)

        # Create or load model
        num_inputs = eval_dataset.inputs.shape[1]
        num_outputs = eval_dataset.targets.shape[1]

        if args.pretrained:
            eval_net = load_net_file(args.pretrained)
            print(f"Loaded pretrained model: {eval_net.num_inputs} -> {eval_net.num_hidden} -> {eval_net.num_outputs}")
        else:
            # Use the architecture factory
            eval_net = get_architecture(args.arch, num_inputs, num_outputs, args.hidden)
            print_model_summary(eval_net, f"Eval Network ({args.arch})")

        # Train
        start_time = time.time()

        if args.overfit_batch is not None:
            # Overfit on batch sanity check
            batch_size = args.overfit_batch if args.overfit_batch > 0 else args.batch_size
            batch_inputs = eval_dataset.inputs[:batch_size]
            batch_targets = eval_dataset.targets[:batch_size]
            history = overfit_batch(
                eval_net, batch_inputs, batch_targets,
                num_steps=args.epochs * 100,  # Use epochs * 100 as step count
                lr=args.lr,
                device=device,
                print_every=100,
            )
        else:
            history = train_eval_network(
                eval_net, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, device=device,
                weight_decay=args.weight_decay,
                scheduler_type=args.scheduler,
                early_stopping_patience=args.early_stopping,
                checkpoint_dir=checkpoint_dir,
                model_name=f'eval_{args.arch}',
                max_examples=args.max_examples,
                use_wandb=use_wandb,
            )
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")

        # Save - only save .net format for baseline (C-compatible format)
        if args.arch == 'baseline':
            output_path = os.path.join(args.output_dir, "eval_trained.net")
            save_net_file(eval_net.cpu(), output_path, getattr(eval_net, 'input_names', None))

        # Save PyTorch format (works for all architectures)
        pt_path = os.path.join(args.output_dir, f"eval_{args.arch}.pt")
        save_dict = {
            'model_state_dict': eval_net.cpu().state_dict(),
            'architecture': args.arch,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'history': history,
        }
        if isinstance(eval_net, ResidualMLP):
            save_dict['config'] = eval_net.get_config()
        torch.save(save_dict, pt_path)
        print(f"Saved model to {pt_path}")

    # Train role network
    if args.network in ["role", "both"]:
        print("\n" + "=" * 60)
        print("TRAINING ROLE NETWORK")
        print("=" * 60)

        # Load data
        print("Loading role training data...")
        role_dataset = RFTGRoleDataset(args.data_dir, max_games=args.max_games)

        # Split into train/val
        val_size = int(len(role_dataset) * args.val_split)
        train_size = len(role_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            role_dataset, [train_size, val_size]
        )

        # Use pin_memory for faster CPU->GPU transfers when using CUDA
        pin_memory = device.type == "cuda"
        num_workers = 4 if device.type == "cuda" else 0
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)

        # Create model - for role network, output is number of possible actions
        num_inputs = role_dataset.inputs.shape[1]
        num_outputs = max(role_dataset.num_actions_list)  # Maximum possible actions

        # Use the architecture factory
        role_net = get_architecture(args.arch, num_inputs, num_outputs, args.hidden)
        print_model_summary(role_net, f"Role Network ({args.arch})")

        # Train
        start_time = time.time()
        history = train_role_network(
            role_net, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            early_stopping_patience=args.early_stopping,
            checkpoint_dir=checkpoint_dir,
            model_name=f'role_{args.arch}',
        )
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")

        # Save - only save .net format for baseline (C-compatible format)
        if args.arch == 'baseline':
            output_path = os.path.join(args.output_dir, "role_trained.net")
            save_net_file(role_net.cpu(), output_path)

        # Save PyTorch format (works for all architectures)
        pt_path = os.path.join(args.output_dir, f"role_{args.arch}.pt")
        save_dict = {
            'model_state_dict': role_net.cpu().state_dict(),
            'architecture': args.arch,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'history': history,
        }
        if isinstance(role_net, ResidualMLP):
            save_dict['config'] = role_net.get_config()
        torch.save(save_dict, pt_path)
        print(f"Saved model to {pt_path}")

    # Train unified network
    if args.network == "unified":
        print("\n" + "=" * 60)
        print("TRAINING UNIFIED NETWORK")
        print("=" * 60)

        # Load combined data
        print("Loading unified training data...")
        unified_dataset = RFTGUnifiedDataset(args.data_dir, max_games=args.max_games)

        # Split into train/val
        val_size = int(len(unified_dataset) * args.val_split)
        train_size = len(unified_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            unified_dataset, [train_size, val_size]
        )

        # Use pin_memory for faster CPU->GPU transfers when using CUDA
        pin_memory = device.type == "cuda"
        num_workers = 4 if device.type == "cuda" else 0
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)

        # Create unified model
        num_inputs = unified_dataset.num_inputs
        num_value_outputs = unified_dataset.num_value_outputs
        num_policy_outputs = unified_dataset.num_policy_outputs

        if not args.arch.startswith('unified-'):
            # Default to unified-medium if no unified arch specified
            args.arch = 'unified-medium'
            print(f"Note: Using {args.arch} architecture for unified training")

        unified_net = get_architecture(
            args.arch,
            num_inputs,
            num_value_outputs,
            num_policy_outputs=num_policy_outputs,
        )
        print_model_summary(unified_net, f"Unified Network ({args.arch})")

        # Train
        start_time = time.time()
        history = train_unified_network(
            unified_net, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            early_stopping_patience=args.early_stopping,
            checkpoint_dir=checkpoint_dir,
            model_name=f'unified_{args.arch.replace("unified-", "")}',
            value_loss_weight=args.value_loss_weight,
            policy_loss_weight=args.policy_loss_weight,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
        )
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")

        # Save PyTorch format
        pt_path = os.path.join(args.output_dir, f"unified_{args.arch.replace('unified-', '')}.pt")
        save_dict = {
            'model_state_dict': unified_net.cpu().state_dict(),
            'architecture': args.arch,
            'num_inputs': num_inputs,
            'num_value_outputs': num_value_outputs,
            'num_policy_outputs': num_policy_outputs,
            'history': history,
        }
        if isinstance(unified_net, UnifiedRFTGNet):
            save_dict['config'] = unified_net.get_config()
        torch.save(save_dict, pt_path)
        print(f"Saved model to {pt_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {args.output_dir}")

    # Finish wandb run
    if use_wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
