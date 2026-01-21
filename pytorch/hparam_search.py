#!/usr/bin/env python3
"""
Hyperparameter optimization for RFTG neural networks using Optuna.

This script searches for optimal hyperparameters for the eval and role networks
using Bayesian optimization with early stopping (pruning) of unpromising trials.

Usage:
    # Search hyperparameters for eval network
    python hparam_search.py -d ./training_data -n eval --n-trials 50

    # Quick test with few trials
    python hparam_search.py -d ./training_data -n eval --n-trials 5

    # Train final model with best hyperparameters
    python hparam_search.py -d ./training_data -n eval --n-trials 50 --train-best
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from train import RFTGEvalDataset, RFTGRoleDataset
from modern_nets import ResidualMLP


def create_model(
    num_inputs: int,
    num_outputs: int,
    hidden_dim: int,
    num_layers: int,
) -> nn.Module:
    """Create a ResidualMLP model with specified architecture."""
    return ResidualMLP(
        num_inputs=num_inputs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_outputs=num_outputs,
    )


def train_epoch_eval(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on eval network. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = F.kl_div(
            torch.log(outputs + 1e-10),
            targets,
            reduction='batchmean'
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples


def validate_eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate eval network. Returns average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = F.kl_div(
                torch.log(outputs + 1e-10),
                targets,
                reduction='batchmean'
            )

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples


def train_epoch_role(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on role network. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples


def validate_role(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Validate role network. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def objective_eval(
    trial: optuna.Trial,
    dataset: RFTGEvalDataset,
    device: torch.device,
    max_epochs: int = 30,
    val_split: float = 0.1,
) -> float:
    """Optuna objective function for eval network hyperparameter search."""

    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    scheduler_type = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none'])

    # Split data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    num_inputs = dataset.inputs.shape[1]
    num_outputs = dataset.targets.shape[1]
    model = create_model(num_inputs, num_outputs, hidden_dim, num_layers)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None

    # Training loop with pruning
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        train_loss = train_epoch_eval(model, train_loader, optimizer, device)
        val_loss = validate_eval(model, val_loader, device)

        # Update scheduler
        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()
        elif scheduler_type == 'plateau' and scheduler:
            scheduler.step(val_loss)

        best_val_loss = min(best_val_loss, val_loss)

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def objective_role(
    trial: optuna.Trial,
    dataset: RFTGRoleDataset,
    device: torch.device,
    max_epochs: int = 30,
    val_split: float = 0.1,
) -> float:
    """Optuna objective function for role network hyperparameter search."""

    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    scheduler_type = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none'])

    # Split data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    num_inputs = dataset.inputs.shape[1]
    num_outputs = max(dataset.num_actions_list)
    model = create_model(num_inputs, num_outputs, hidden_dim, num_layers)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None

    # Training loop with pruning
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        train_loss = train_epoch_role(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_role(model, val_loader, device)

        # Update scheduler
        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()
        elif scheduler_type == 'plateau' and scheduler:
            scheduler.step(val_loss)

        best_val_loss = min(best_val_loss, val_loss)

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def run_hyperparameter_search(
    data_dirs: list,
    network_type: str,
    n_trials: int,
    device: torch.device,
    max_epochs: int = 30,
    max_games: Optional[int] = None,
    output_dir: str = './hparam_results',
) -> Dict[str, Any]:
    """
    Run hyperparameter search using Optuna.

    Returns:
        Dictionary with best hyperparameters and study results
    """
    print(f"Loading {network_type} training data...")

    if network_type == 'eval':
        dataset = RFTGEvalDataset(data_dirs, max_games=max_games)
        objective_fn = lambda trial: objective_eval(trial, dataset, device, max_epochs)
    else:
        dataset = RFTGRoleDataset(data_dirs, max_games=max_games)
        objective_fn = lambda trial: objective_role(trial, dataset, device, max_epochs)

    print(f"Loaded {len(dataset)} samples")
    print(f"Starting hyperparameter search with {n_trials} trials...")
    print()

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # Run optimization
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print results
    print()
    print("=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_value:.6f}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'network_type': network_type,
        'n_trials': n_trials,
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }

    results_path = os.path.join(output_dir, f'{network_type}_hparam_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


def train_with_best_params(
    data_dirs: list,
    network_type: str,
    params: Dict[str, Any],
    device: torch.device,
    max_epochs: int = 50,
    max_games: Optional[int] = None,
    output_dir: str = './trained_models',
    val_split: float = 0.1,
) -> None:
    """Train a model with the best hyperparameters found."""
    print(f"Training {network_type} network with best hyperparameters...")
    print(f"Params: {params}")
    print()

    # Load data
    if network_type == 'eval':
        dataset = RFTGEvalDataset(data_dirs, max_games=max_games)
        num_inputs = dataset.inputs.shape[1]
        num_outputs = dataset.targets.shape[1]
    else:
        dataset = RFTGRoleDataset(data_dirs, max_games=max_games)
        num_inputs = dataset.inputs.shape[1]
        num_outputs = max(dataset.num_actions_list)

    # Split data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = create_model(num_inputs, num_outputs, params['hidden_dim'], params['num_layers'])
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    scheduler_type = params['scheduler']
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=params['lr'] * 0.01)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(max_epochs):
        if network_type == 'eval':
            train_loss = train_epoch_eval(model, train_loader, optimizer, device)
            val_loss = validate_eval(model, val_loader, device)
            print(f"Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                  f"{' *' if val_loss < best_val_loss else ''}")
        else:
            train_loss = train_epoch_role(model, train_loader, optimizer, device)
            val_loss, val_acc = validate_role(model, val_loader, device)
            print(f"Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                  f"{' *' if val_loss < best_val_loss else ''}")

        # Update scheduler
        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()
        elif scheduler_type == 'plateau' and scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{network_type}_optimized.pt')
    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'params': params,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'best_val_loss': best_val_loss,
    }, model_path)
    print(f"\nModel saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for RFTG neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        nargs='+',
        required=True,
        help="Directory/directories containing training data"
    )
    parser.add_argument(
        "--network", "-n",
        type=str,
        choices=['eval', 'role'],
        required=True,
        help="Network type to optimize"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max epochs per trial (default: 30)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Max games to load (for testing)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./hparam_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--train-best",
        action="store_true",
        help="Train final model with best hyperparameters"
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=50,
        help="Epochs for final training with best params (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Run hyperparameter search
    results = run_hyperparameter_search(
        data_dirs=args.data_dir,
        network_type=args.network,
        n_trials=args.n_trials,
        device=device,
        max_epochs=args.max_epochs,
        max_games=args.max_games,
        output_dir=args.output_dir,
    )

    # Train with best params if requested
    if args.train_best:
        print()
        print("=" * 60)
        print("TRAINING WITH BEST HYPERPARAMETERS")
        print("=" * 60)
        train_with_best_params(
            data_dirs=args.data_dir,
            network_type=args.network,
            params=results['best_params'],
            device=device,
            max_epochs=args.train_epochs,
            max_games=args.max_games,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
