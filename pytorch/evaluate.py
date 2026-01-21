#!/usr/bin/env python3
"""
Model evaluation and comparison tool for RFTG neural networks.

Evaluates trained models on held-out data and computes various metrics
to compare different architectures.

Metrics computed:
1. KL Divergence Loss - Primary metric (lower = better)
2. Accuracy - Does predicted winner match actual winner?
3. Calibration - Are predicted probabilities well-calibrated?
4. Per-round Analysis - Loss by game round (early/mid/late game)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from rftg_net import RFTGNet, load_net_file
from modern_nets import get_architecture, ResidualMLP


class RFTGEvalDatasetWithRound(Dataset):
    """Dataset for eval network that also tracks round numbers."""

    def __init__(self, data_dir: str, max_games: Optional[int] = None):
        self.inputs = []
        self.targets = []
        self.rounds = []  # Track round numbers for per-round analysis

        games_loaded = 0
        data_path = Path(data_dir)

        for filepath in sorted(data_path.glob("*.jsonl")):
            with open(filepath, 'r') as f:
                for line in f:
                    if max_games and games_loaded >= max_games:
                        break
                    try:
                        game = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    num_players = game['num_players']
                    winners = set(game['winner_indices'])
                    target = [1.0 / len(winners) if i in winners else 0.0
                              for i in range(num_players)]

                    for state in game['eval_states']:
                        self.inputs.append(state['inputs'])
                        player_idx = state['player_index']
                        rotated_target = target[player_idx:] + target[:player_idx]
                        self.targets.append(rotated_target)
                        self.rounds.append(state.get('round_num', 0))

                    games_loaded += 1

            if max_games and games_loaded >= max_games:
                break

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.rounds = torch.tensor(self.rounds, dtype=torch.long)

        print(f"Loaded {len(self.inputs)} eval states from {games_loaded} games")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.rounds[idx]


def load_model(model_path: str) -> Tuple[nn.Module, str]:
    """
    Load a model from either .net or .pt format.

    Args:
        model_path: Path to model file

    Returns:
        Tuple of (model, architecture_name)
    """
    path = Path(model_path)

    if path.suffix == '.net':
        # Load C-format network (baseline only)
        model = load_net_file(str(path))
        return model, 'baseline'

    elif path.suffix == '.pt':
        # Load PyTorch checkpoint
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'architecture' in checkpoint:
            arch = checkpoint['architecture']
            num_inputs = checkpoint['num_inputs']
            num_outputs = checkpoint['num_outputs']

            model = get_architecture(arch, num_inputs, num_outputs)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            return model, arch
        else:
            # Old format - assume state dict only, try to infer architecture
            # This is a fallback for simpler checkpoints
            raise ValueError(f"Cannot determine architecture from checkpoint: {model_path}")

    else:
        raise ValueError(f"Unknown model format: {path.suffix}")


def compute_kl_divergence(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute KL divergence loss."""
    return F.kl_div(
        torch.log(outputs + 1e-10),
        targets,
        reduction='batchmean'
    ).item()


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy: does the predicted winner match the actual winner?

    For multi-player games, checks if the player with highest predicted
    probability actually won.
    """
    pred_winner = outputs.argmax(dim=1)
    actual_winner = targets.argmax(dim=1)
    return (pred_winner == actual_winner).float().mean().item()


def compute_calibration(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics.

    Divides predictions into bins by confidence and measures how
    well predicted probabilities match actual outcomes.

    Returns:
        Dictionary with:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - bin_accuracies: Accuracy per bin
        - bin_confidences: Average confidence per bin
        - bin_counts: Number of samples per bin
    """
    # Get predicted probabilities for the player at index 0 (self)
    confidences = outputs[:, 0].cpu().numpy()
    actuals = (targets[:, 0] > 0.5).cpu().numpy().astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accuracies.append(actuals[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error
    weights = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else bin_counts
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))

    # Maximum Calibration Error
    mce = np.max(np.abs(bin_accuracies - bin_confidences))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies.tolist(),
        'bin_confidences': bin_confidences.tolist(),
        'bin_counts': bin_counts.tolist(),
    }


def compute_per_round_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    rounds: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute loss broken down by game round.

    Returns:
        Dictionary with losses for early (0-3), mid (4-7), late (8+) game
    """
    results = {}

    # Early game: rounds 0-3
    early_mask = rounds <= 3
    if early_mask.sum() > 0:
        early_loss = F.kl_div(
            torch.log(outputs[early_mask] + 1e-10),
            targets[early_mask],
            reduction='batchmean'
        ).item()
        results['early_game_loss'] = early_loss
        results['early_game_count'] = int(early_mask.sum())

    # Mid game: rounds 4-7
    mid_mask = (rounds >= 4) & (rounds <= 7)
    if mid_mask.sum() > 0:
        mid_loss = F.kl_div(
            torch.log(outputs[mid_mask] + 1e-10),
            targets[mid_mask],
            reduction='batchmean'
        ).item()
        results['mid_game_loss'] = mid_loss
        results['mid_game_count'] = int(mid_mask.sum())

    # Late game: rounds 8+
    late_mask = rounds >= 8
    if late_mask.sum() > 0:
        late_loss = F.kl_div(
            torch.log(outputs[late_mask] + 1e-10),
            targets[late_mask],
            reduction='batchmean'
        ).item()
        results['late_game_loss'] = late_loss
        results['late_game_count'] = int(late_mask.sum())

    return results


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Neural network model
        dataloader: DataLoader providing (inputs, targets, rounds)
        device: Device to run evaluation on

    Returns:
        Dictionary with all computed metrics
    """
    model = model.to(device)
    model.eval()

    all_outputs = []
    all_targets = []
    all_rounds = []

    with torch.no_grad():
        for inputs, targets, rounds in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets)
            all_rounds.append(rounds)

    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    rounds = torch.cat(all_rounds, dim=0)

    # Compute metrics
    results = {
        'kl_divergence': compute_kl_divergence(outputs, targets),
        'accuracy': compute_accuracy(outputs, targets),
        'num_samples': len(outputs),
    }

    # Calibration
    calibration = compute_calibration(outputs, targets)
    results['calibration_ece'] = calibration['ece']
    results['calibration_mce'] = calibration['mce']

    # Per-round analysis
    per_round = compute_per_round_loss(outputs, targets, rounds)
    results.update(per_round)

    return results


def compare_models(
    model_paths: List[str],
    data_dir: str,
    max_games: Optional[int] = None,
    batch_size: int = 512,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same dataset.

    Args:
        model_paths: List of paths to model files
        data_dir: Directory containing evaluation data
        max_games: Maximum games to load
        batch_size: Batch size for evaluation

    Returns:
        Dictionary mapping model names to their metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading evaluation data...")
    dataset = RFTGEvalDatasetWithRound(data_dir, max_games=max_games)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = {}

    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")

        try:
            model, arch = load_model(model_path)
            model_name = Path(model_path).stem

            # Add parameter count
            param_count = sum(p.numel() for p in model.parameters())

            metrics = evaluate_model(model, dataloader, device)
            metrics['architecture'] = arch
            metrics['parameters'] = param_count
            metrics['model_path'] = model_path

            results[model_name] = metrics
            print(f"  KL Divergence: {metrics['kl_divergence']:.6f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Parameters: {param_count:,}")

        except Exception as e:
            print(f"  Error: {e}")
            results[model_path] = {'error': str(e)}

    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON RESULTS")
    print("=" * 100)

    # Filter out error results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if not valid_results:
        print("No valid results to display")
        return

    # Header
    print(f"{'Model':<30} {'Arch':<15} {'Params':>12} {'KL Div':>10} {'Acc':>8} {'ECE':>8}")
    print("-" * 100)

    # Sort by KL divergence (best first)
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['kl_divergence'])

    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} "
              f"{metrics['architecture']:<15} "
              f"{metrics['parameters']:>12,} "
              f"{metrics['kl_divergence']:>10.6f} "
              f"{metrics['accuracy']:>8.4f} "
              f"{metrics['calibration_ece']:>8.4f}")

    print("-" * 100)

    # Per-round analysis if available
    if 'early_game_loss' in list(valid_results.values())[0]:
        print("\nPER-ROUND LOSS ANALYSIS")
        print("-" * 80)
        print(f"{'Model':<30} {'Early (0-3)':>15} {'Mid (4-7)':>15} {'Late (8+)':>15}")
        print("-" * 80)

        for model_name, metrics in sorted_results:
            early = metrics.get('early_game_loss', float('nan'))
            mid = metrics.get('mid_game_loss', float('nan'))
            late = metrics.get('late_game_loss', float('nan'))
            print(f"{model_name:<30} {early:>15.6f} {mid:>15.6f} {late:>15.6f}")

    print("=" * 100)

    # Summary
    best_model = sorted_results[0]
    worst_model = sorted_results[-1]
    improvement = (worst_model[1]['kl_divergence'] - best_model[1]['kl_divergence']) / worst_model[1]['kl_divergence'] * 100

    print(f"\nBest model: {best_model[0]} (KL={best_model[1]['kl_divergence']:.6f})")
    if len(sorted_results) > 1:
        print(f"Improvement over worst: {improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare RFTG neural network models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model
  python evaluate.py --models eval_baseline.pt -d /tmp/rftg_training_5min

  # Compare multiple models
  python evaluate.py --models baseline.pt small.pt medium.pt -d ./data

  # Evaluate with limited data for quick testing
  python evaluate.py --models *.pt -d ./data --max-games 1000
        """
    )
    parser.add_argument("--models", "-m", nargs="+", required=True,
                        help="Model files to evaluate (.net or .pt format)")
    parser.add_argument("--data-dir", "-d", type=str, required=True,
                        help="Directory containing evaluation data (*.jsonl)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Maximum games to load")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for evaluation")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Expand glob patterns
    import glob as glob_module
    model_paths = []
    for pattern in args.models:
        matches = glob_module.glob(pattern)
        if matches:
            model_paths.extend(matches)
        else:
            model_paths.append(pattern)

    print(f"Evaluating {len(model_paths)} model(s)")

    results = compare_models(
        model_paths,
        args.data_dir,
        max_games=args.max_games,
        batch_size=args.batch_size,
    )

    print_comparison_table(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
