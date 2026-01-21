#!/usr/bin/env python3
"""
Competitive mode for RFTG neural networks.

Pits different trained models against each other by temporarily swapping
network files and running games through the C learner.

Usage:
    python compete.py --model-a trained_models/eval_baseline.pt \
                      --model-b ../asset/network/rftg.eval.0.2.net \
                      --games 100
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import torch
import torch.nn as nn

from rftg_net import RFTGNet, load_net_file
from modern_nets import get_architecture, ResidualMLP


def find_learner_binary() -> Optional[Path]:
    """Find the learner binary."""
    candidates = [
        Path(__file__).parent.parent / "build" / "bin" / "learner",
        Path(__file__).parent.parent / "build" / "learner",
        Path("/home/rrenaud/rftg/build/bin/learner"),
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def find_asset_dir() -> Optional[Path]:
    """Find the asset directory containing cards.txt."""
    candidates = [
        Path(__file__).parent.parent / "asset",
        Path("/home/rrenaud/rftg/asset"),
    ]
    for path in candidates:
        if (path / "cards.txt").exists():
            return path
    return None


def load_model(model_path: str) -> Tuple[nn.Module, str]:
    """Load a model from either .net or .pt format."""
    path = Path(model_path)

    if path.suffix == '.net':
        model = load_net_file(str(path))
        return model, 'baseline'

    elif path.suffix == '.pt':
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)

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
            raise ValueError(f"Cannot determine architecture from checkpoint: {model_path}")

    else:
        raise ValueError(f"Unknown model format: {path.suffix}")


def save_model_as_net(model: nn.Module, filepath: str, input_names: Optional[List[str]] = None):
    """
    Save a PyTorch model to .net file format compatible with the C code.

    Only works for baseline RFTGNet architecture (2-layer MLP).
    """
    if not isinstance(model, RFTGNet):
        raise ValueError("Can only save RFTGNet (baseline) models to .net format")

    if input_names is None:
        if hasattr(model, 'input_names') and model.input_names:
            input_names = model.input_names
        else:
            input_names = [f"input_{i}" for i in range(model.num_inputs)]

    with open(filepath, 'w') as f:
        f.write(f"{model.num_inputs} {model.num_hidden} {model.num_outputs}\n")
        f.write(f"{getattr(model, 'num_training', 0)}\n")

        for i in range(model.num_inputs):
            if i < len(input_names):
                f.write(f"{input_names[i]}\n")
            else:
                f.write(f"input_{i}\n")

        hidden_weight = model.hidden.weight.detach().cpu().numpy()
        hidden_bias = model.hidden.bias.detach().cpu().numpy()

        for i in range(model.num_hidden):
            for j in range(model.num_inputs):
                f.write(f"{hidden_weight[i, j]}\n")
            f.write(f"{hidden_bias[i]}\n")

        output_weight = model.output.weight.detach().cpu().numpy()
        output_bias = model.output.bias.detach().cpu().numpy()

        for i in range(model.num_outputs):
            for j in range(model.num_hidden):
                f.write(f"{output_weight[i, j]}\n")
            f.write(f"{output_bias[i]}\n")


def convert_to_baseline(model: nn.Module, arch: str, num_hidden: int = 50) -> RFTGNet:
    """
    Convert a model to baseline format by distillation (if needed).

    For non-baseline models, this creates a baseline model that approximates
    the larger model's behavior through knowledge distillation on random inputs.
    """
    if isinstance(model, RFTGNet):
        return model

    # For ResidualMLP or other architectures, we need to distill
    # Get input/output dimensions
    if hasattr(model, 'num_inputs'):
        num_inputs = model.num_inputs
    elif hasattr(model, 'input_dim'):
        num_inputs = model.input_dim
    else:
        # Try to infer from first layer
        for name, param in model.named_parameters():
            if 'weight' in name:
                num_inputs = param.shape[1]
                break

    if hasattr(model, 'num_outputs'):
        num_outputs = model.num_outputs
    elif hasattr(model, 'output_dim'):
        num_outputs = model.output_dim
    else:
        num_outputs = 2  # Default for eval network

    print(f"  Distilling {arch} model to baseline ({num_inputs} -> {num_hidden} -> {num_outputs})")

    # Create baseline model
    baseline = RFTGNet(num_inputs, num_hidden, num_outputs)

    # Distill: train baseline to match the larger model
    model.eval()
    baseline.train()

    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)

    # Generate random training data and distill
    num_samples = 50000
    batch_size = 512

    with torch.no_grad():
        # Generate random inputs (normalized to typical game state ranges)
        inputs = torch.rand(num_samples, num_inputs) * 2 - 0.5
        targets = model(inputs)

    # Train baseline to match
    for epoch in range(20):
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            optimizer.zero_grad()
            outputs = baseline(batch_inputs)

            # KL divergence loss
            loss = torch.nn.functional.kl_div(
                torch.log(outputs + 1e-10),
                batch_targets,
                reduction='batchmean'
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Distillation epoch {epoch+1}: loss={total_loss / (num_samples // batch_size):.6f}")

    baseline.eval()
    return baseline


def run_games(
    learner_path: Path,
    asset_dir: Path,
    num_games: int,
    num_players: int = 2,
    expansion: int = 0,
    advanced: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run games and return results."""
    cmd = [
        str(learner_path),
        "-n", str(num_games),
        "-p", str(num_players),
        "-e", str(expansion),
        "-r", str(seed),
        "-f", "0.0",  # No training
    ]

    if advanced:
        cmd.append("-a")

    if verbose:
        cmd.append("-v")

    # Create temp file for export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        export_file = f.name

    cmd.extend(["-x", export_file])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(asset_dir)
        )

        # Parse results from export file
        wins = [0] * num_players
        total_games = 0

        if os.path.exists(export_file):
            with open(export_file, 'r') as f:
                for line in f:
                    try:
                        game = json.loads(line)
                        total_games += 1
                        for winner_idx in game.get('winner_indices', []):
                            wins[winner_idx] += 1
                    except json.JSONDecodeError:
                        continue

        return {
            'success': result.returncode == 0,
            'total_games': total_games,
            'wins': wins,
            'win_rates': [w / total_games if total_games > 0 else 0 for w in wins],
            'stderr': result.stderr if result.returncode != 0 else '',
        }

    finally:
        if os.path.exists(export_file):
            os.remove(export_file)


def compete(
    model_a_path: str,
    model_b_path: str,
    num_games: int = 100,
    num_players: int = 2,
    expansion: int = 0,
    advanced: bool = False,
    hidden_size: int = 50,
    seed: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a competition between two models.

    Runs games with model_a as player 0, then swaps and runs with model_b as player 0.
    Returns aggregate statistics.
    """
    learner_path = find_learner_binary()
    if not learner_path:
        raise FileNotFoundError("Could not find learner binary")

    asset_dir = find_asset_dir()
    if not asset_dir:
        raise FileNotFoundError("Could not find asset directory")

    if seed is None:
        seed = int(time.time())

    # Network file paths
    eval_net_path = asset_dir / "network" / f"rftg.eval.{expansion}.{num_players}{'a' if advanced else ''}.net"
    role_net_path = asset_dir / "network" / f"rftg.role.{expansion}.{num_players}{'a' if advanced else ''}.net"

    # Backup original networks
    eval_backup = str(eval_net_path) + ".backup"
    role_backup = str(role_net_path) + ".backup"

    print(f"Network path: {eval_net_path}")

    # Load models
    print(f"\nLoading Model A: {model_a_path}")
    model_a, arch_a = load_model(model_a_path)
    print(f"  Architecture: {arch_a}")

    print(f"\nLoading Model B: {model_b_path}")
    model_b, arch_b = load_model(model_b_path)
    print(f"  Architecture: {arch_b}")

    # Convert to baseline format if needed
    print("\nPreparing models for competition...")
    baseline_a = convert_to_baseline(model_a, arch_a, hidden_size)
    baseline_b = convert_to_baseline(model_b, arch_b, hidden_size)

    # Get input names from original network
    original_net = load_net_file(str(eval_net_path))
    input_names = original_net.input_names

    results = {
        'model_a': model_a_path,
        'model_b': model_b_path,
        'arch_a': arch_a,
        'arch_b': arch_b,
        'num_games': num_games,
        'games_per_side': num_games // 2,
    }

    try:
        # Backup original networks
        shutil.copy(eval_net_path, eval_backup)
        if role_net_path.exists():
            shutil.copy(role_net_path, role_backup)

        # Round 1: Model A as the network (all players use it)
        # This tests Model A's overall strength
        print(f"\n{'='*60}")
        print(f"Round 1: Testing Model A ({num_games // 2} games)")
        print(f"{'='*60}")

        save_model_as_net(baseline_a, str(eval_net_path), input_names)

        round1 = run_games(
            learner_path, asset_dir,
            num_games=num_games // 2,
            num_players=num_players,
            expansion=expansion,
            advanced=advanced,
            seed=seed,
            verbose=verbose,
        )

        print(f"Games completed: {round1['total_games']}")
        print(f"Player win rates: {[f'{r:.1%}' for r in round1['win_rates']]}")

        # Round 2: Model B as the network
        print(f"\n{'='*60}")
        print(f"Round 2: Testing Model B ({num_games // 2} games)")
        print(f"{'='*60}")

        save_model_as_net(baseline_b, str(eval_net_path), input_names)

        round2 = run_games(
            learner_path, asset_dir,
            num_games=num_games // 2,
            num_players=num_players,
            expansion=expansion,
            advanced=advanced,
            seed=seed + 1000000,  # Different seed for variety
            verbose=verbose,
        )

        print(f"Games completed: {round2['total_games']}")
        print(f"Player win rates: {[f'{r:.1%}' for r in round2['win_rates']]}")

        results['round1'] = round1
        results['round2'] = round2

        # Analysis: Compare the networks by looking at how well they play
        # Both networks are used for all players (self-play), so we compare
        # the quality of play indirectly through game statistics

    finally:
        # Restore original networks
        if os.path.exists(eval_backup):
            shutil.move(eval_backup, eval_net_path)
        if os.path.exists(role_backup):
            shutil.move(role_backup, role_net_path)

    return results


def decision_comparison(
    model_a_path: str,
    model_b_path: str,
    data_dir: str,
    max_games: int = 1000,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compare models on the same game states from training data.

    Evaluates which model's predictions better match actual game outcomes.
    """
    from train import RFTGEvalDataset

    print(f"\nLoading Model A: {model_a_path}")
    model_a, arch_a = load_model(model_a_path)
    print(f"  Architecture: {arch_a}")

    print(f"\nLoading Model B: {model_b_path}")
    model_b, arch_b = load_model(model_b_path)
    print(f"  Architecture: {arch_b}")

    print(f"\nLoading evaluation data from {data_dir}...")
    dataset = RFTGEvalDataset(data_dir, max_games=max_games)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {dev}")

    model_a = model_a.to(dev).eval()
    model_b = model_b.to(dev).eval()

    inputs = dataset.inputs.to(dev)
    targets = dataset.targets.to(dev)

    with torch.no_grad():
        preds_a = model_a(inputs)
        preds_b = model_b(inputs)

    # KL divergence (lower = better)
    kl_a = torch.nn.functional.kl_div(
        torch.log(preds_a + 1e-10), targets, reduction='batchmean'
    ).item()
    kl_b = torch.nn.functional.kl_div(
        torch.log(preds_b + 1e-10), targets, reduction='batchmean'
    ).item()

    # Accuracy (higher = better)
    pred_winners_a = preds_a.argmax(dim=1)
    pred_winners_b = preds_b.argmax(dim=1)
    actual_winners = targets.argmax(dim=1)

    acc_a = (pred_winners_a == actual_winners).float().mean().item()
    acc_b = (pred_winners_b == actual_winners).float().mean().item()

    # Head-to-head: on which samples does each model make better predictions?
    # Use log probability of actual outcome as the metric
    log_prob_a = torch.log(preds_a + 1e-10).gather(1, actual_winners.unsqueeze(1)).squeeze()
    log_prob_b = torch.log(preds_b + 1e-10).gather(1, actual_winners.unsqueeze(1)).squeeze()

    a_better = (log_prob_a > log_prob_b).sum().item()
    b_better = (log_prob_b > log_prob_a).sum().item()
    ties = (log_prob_a == log_prob_b).sum().item()

    # Agreement: how often do they predict the same winner?
    agreement = (pred_winners_a == pred_winners_b).float().mean().item()

    results = {
        'model_a': model_a_path,
        'model_b': model_b_path,
        'arch_a': arch_a,
        'arch_b': arch_b,
        'num_samples': len(inputs),
        'kl_a': kl_a,
        'kl_b': kl_b,
        'accuracy_a': acc_a,
        'accuracy_b': acc_b,
        'a_wins': a_better,
        'b_wins': b_better,
        'ties': ties,
        'agreement': agreement,
    }

    return results


def print_decision_results(results: Dict[str, Any]):
    """Print decision comparison results."""
    print(f"\n{'='*60}")
    print("DECISION COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Model A: {Path(results['model_a']).name} ({results['arch_a']})")
    print(f"Model B: {Path(results['model_b']).name} ({results['arch_b']})")
    print(f"Samples evaluated: {results['num_samples']:,}")
    print()

    print("Prediction Quality (lower KL = better, higher accuracy = better):")
    print(f"  Model A: KL={results['kl_a']:.6f}, Accuracy={results['accuracy_a']:.1%}")
    print(f"  Model B: KL={results['kl_b']:.6f}, Accuracy={results['accuracy_b']:.1%}")
    print()

    total = results['a_wins'] + results['b_wins'] + results['ties']
    print("Head-to-Head (which model assigns higher probability to actual winner):")
    print(f"  Model A better: {results['a_wins']:,} ({results['a_wins']/total:.1%})")
    print(f"  Model B better: {results['b_wins']:,} ({results['b_wins']/total:.1%})")
    print(f"  Ties: {results['ties']:,} ({results['ties']/total:.1%})")
    print()

    print(f"Prediction Agreement: {results['agreement']:.1%}")
    print()

    # Declare winner
    if results['kl_a'] < results['kl_b'] and results['a_wins'] > results['b_wins']:
        winner = "Model A"
    elif results['kl_b'] < results['kl_a'] and results['b_wins'] > results['a_wins']:
        winner = "Model B"
    else:
        winner = "Mixed results"

    print(f"Overall: {winner}")
    print(f"{'='*60}")


def print_results(results: Dict[str, Any]):
    """Print competition results."""
    print(f"\n{'='*60}")
    print("COMPETITION RESULTS")
    print(f"{'='*60}")
    print(f"Model A: {Path(results['model_a']).name} ({results['arch_a']})")
    print(f"Model B: {Path(results['model_b']).name} ({results['arch_b']})")
    print(f"Games per model: {results['games_per_side']}")
    print()

    # Since both are self-play (all players use same network),
    # we can compare the balance/quality of play
    r1 = results['round1']
    r2 = results['round2']

    print("Win Rate Distribution (balanced = 50% each for 2 players):")
    print(f"  Model A: Player 0: {r1['win_rates'][0]:.1%}, Player 1: {r1['win_rates'][1]:.1%}")
    print(f"  Model B: Player 0: {r2['win_rates'][0]:.1%}, Player 1: {r2['win_rates'][1]:.1%}")

    # Compute balance metric (how close to 50/50)
    balance_a = 1 - abs(r1['win_rates'][0] - 0.5) * 2
    balance_b = 1 - abs(r2['win_rates'][0] - 0.5) * 2

    print(f"\nBalance Score (1.0 = perfectly balanced):")
    print(f"  Model A: {balance_a:.3f}")
    print(f"  Model B: {balance_b:.3f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run competitive matches between RFTG models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decision comparison (recommended) - compares on same game states
  python compete.py -a trained_models/eval_baseline.pt -b ../asset/network/rftg.eval.0.2.net --mode decision -d training_data/batch_20260119_181046

  # Self-play comparison - tests each model's balance in self-play
  python compete.py -a model1.pt -b model2.pt --mode selfplay --games 100
"""
    )
    parser.add_argument("--model-a", "-a", required=True,
                        help="Path to first model (.net or .pt)")
    parser.add_argument("--model-b", "-b", required=True,
                        help="Path to second model (.net or .pt)")
    parser.add_argument("--mode", "-m", choices=["decision", "selfplay"], default="decision",
                        help="Competition mode: 'decision' (compare on same states) or 'selfplay' (default: decision)")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                        help="Data directory for decision mode (required for decision mode)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max games to load for decision mode")
    parser.add_argument("--games", "-n", type=int, default=100,
                        help="Total number of games for selfplay mode (default: 100)")
    parser.add_argument("--players", "-p", type=int, default=2,
                        help="Number of players (default: 2)")
    parser.add_argument("--expansion", "-e", type=int, default=0,
                        help="Expansion level (default: 0)")
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced 2-player rules")
    parser.add_argument("--hidden", type=int, default=50,
                        help="Hidden layer size for distillation (default: 50)")
    parser.add_argument("--seed", "-r", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for decision mode (default: cuda)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    try:
        if args.mode == "decision":
            if not args.data_dir:
                print("Error: --data-dir is required for decision mode")
                return 1

            results = decision_comparison(
                model_a_path=args.model_a,
                model_b_path=args.model_b,
                data_dir=args.data_dir,
                max_games=args.max_games,
                device=args.device,
            )
            print_decision_results(results)

        else:  # selfplay mode
            results = compete(
                model_a_path=args.model_a,
                model_b_path=args.model_b,
                num_games=args.games,
                num_players=args.players,
                expansion=args.expansion,
                advanced=args.advanced,
                hidden_size=args.hidden,
                seed=args.seed,
                verbose=args.verbose,
            )
            print_results(results)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
