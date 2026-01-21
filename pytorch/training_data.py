"""
Training data structures and I/O for RFTG neural network learning.

This module defines the data format for capturing self-play games
and provides utilities for reading/writing training data.

Training Approach (from C implementation):
- Eval network: TD(λ) learning with λ=0.7 decay
  - Input: game state features
  - Target: win probability (1 for winner, 0 for losers, or proportional to scores)

- Role network: Policy gradient-style
  - Input: game state features
  - Target: softmax of action scores (from Monte Carlo rollouts)
"""

import json
import gzip
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import struct
import io


@dataclass
class EvalState:
    """
    A single game state for eval network training.

    The eval network predicts win probability for each player.
    States are collected during gameplay, then trained with TD(λ)
    at game end using the actual outcome as the final target.
    """
    player_index: int          # Which player's perspective (0 to num_players-1)
    round_num: int             # Game round when this state was recorded
    inputs: np.ndarray         # Feature vector, shape (num_inputs,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_index': self.player_index,
            'round_num': self.round_num,
            'inputs': self.inputs.tolist()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvalState':
        return cls(
            player_index=d['player_index'],
            round_num=d['round_num'],
            inputs=np.array(d['inputs'], dtype=np.float32)
        )


@dataclass
class RoleDecision:
    """
    A single action decision for role network training.

    The role network predicts which action(s) a player will choose.
    Training target is softmax of action scores from MCTS-style evaluation.
    """
    player_index: int          # Which player made this decision
    round_num: int             # Game round
    inputs: np.ndarray         # Feature vector for role network
    chosen_action: int         # Index of action actually chosen
    action_scores: np.ndarray  # Scores for all possible actions (from rollouts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'player_index': self.player_index,
            'round_num': self.round_num,
            'inputs': self.inputs.tolist(),
            'chosen_action': self.chosen_action,
            'action_scores': self.action_scores.tolist()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RoleDecision':
        return cls(
            player_index=d['player_index'],
            round_num=d['round_num'],
            inputs=np.array(d['inputs'], dtype=np.float32),
            chosen_action=d['chosen_action'],
            action_scores=np.array(d['action_scores'], dtype=np.float32)
        )


@dataclass
class GameRecord:
    """
    Complete record of a single game for training.

    Contains all the data needed to train both eval and role networks.
    """
    # Game configuration
    game_id: str               # Unique identifier (e.g., "{seed}_{timestamp}")
    expansion: int             # Expansion level (0=base, 1=GS, 2=RvI, 3=BoW, 4=AA)
    num_players: int           # Number of players (2-6)
    advanced: bool             # Advanced 2-player variant
    random_seed: int           # Starting random seed

    # Game outcome
    winner_indices: List[int]  # Indices of winning player(s) - can be multiple in tie
    final_scores: List[int]    # Final VP scores for each player

    # Training data
    eval_states: List[EvalState] = field(default_factory=list)
    role_decisions: List[RoleDecision] = field(default_factory=list)

    # Metadata
    num_rounds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'game_id': self.game_id,
            'expansion': self.expansion,
            'num_players': self.num_players,
            'advanced': self.advanced,
            'random_seed': self.random_seed,
            'winner_indices': self.winner_indices,
            'final_scores': self.final_scores,
            'eval_states': [s.to_dict() for s in self.eval_states],
            'role_decisions': [d.to_dict() for d in self.role_decisions],
            'num_rounds': self.num_rounds
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GameRecord':
        return cls(
            game_id=d['game_id'],
            expansion=d['expansion'],
            num_players=d['num_players'],
            advanced=d['advanced'],
            random_seed=d['random_seed'],
            winner_indices=d['winner_indices'],
            final_scores=d['final_scores'],
            eval_states=[EvalState.from_dict(s) for s in d['eval_states']],
            role_decisions=[RoleDecision.from_dict(r) for r in d['role_decisions']],
            num_rounds=d.get('num_rounds', 0)
        )

    def get_eval_targets(self, method: str = 'winner') -> np.ndarray:
        """
        Compute training targets for eval network.

        Args:
            method: 'winner' (1 for winner, 0 for losers) or
                    'score' (proportional to final scores)

        Returns:
            Array of shape (num_players,) with target probabilities
        """
        targets = np.zeros(self.num_players, dtype=np.float32)

        if method == 'winner':
            # Binary: winners get equal share of 1.0
            for i in self.winner_indices:
                targets[i] = 1.0 / len(self.winner_indices)
        elif method == 'score':
            # Softmax of scores
            scores = np.array(self.final_scores, dtype=np.float32)
            exp_scores = np.exp(scores - scores.max())  # Numerical stability
            targets = exp_scores / exp_scores.sum()
        else:
            raise ValueError(f"Unknown method: {method}")

        return targets


class TrainingDataWriter:
    """
    Writes training data to disk in an efficient format.

    Format: gzipped JSON lines (one game per line)
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self):
        self._file = gzip.open(self.filepath, 'wt', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def write_game(self, game: GameRecord):
        """Write a single game record."""
        json.dump(game.to_dict(), self._file)
        self._file.write('\n')

    def write_games(self, games: List[GameRecord]):
        """Write multiple game records."""
        for game in games:
            self.write_game(game)


class TrainingDataReader:
    """
    Reads training data from disk.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def __iter__(self):
        """Iterate over game records."""
        with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield GameRecord.from_dict(json.loads(line))

    def read_all(self) -> List[GameRecord]:
        """Read all game records into memory."""
        return list(self)

    def count_games(self) -> int:
        """Count number of games without loading all data."""
        count = 0
        with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class TrainingBatch:
    """
    A batch of training data ready for PyTorch.

    Collates multiple games into tensors for batch training.
    """

    def __init__(self, games: List[GameRecord], network_type: str = 'eval'):
        """
        Create a training batch from game records.

        Args:
            games: List of game records
            network_type: 'eval' or 'role'
        """
        self.network_type = network_type

        if network_type == 'eval':
            self._build_eval_batch(games)
        elif network_type == 'role':
            self._build_role_batch(games)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def _build_eval_batch(self, games: List[GameRecord]):
        """Build batch for eval network training."""
        inputs_list = []
        targets_list = []
        lambdas_list = []  # TD(λ) weights

        for game in games:
            targets = game.get_eval_targets('winner')

            # Process states in reverse order (most recent first) with λ decay
            # Group by player
            player_states = {}
            for state in game.eval_states:
                if state.player_index not in player_states:
                    player_states[state.player_index] = []
                player_states[state.player_index].append(state)

            # For each player, add states with decaying lambda
            for player_idx, states in player_states.items():
                lambda_val = 1.0
                for state in reversed(states):
                    inputs_list.append(state.inputs)
                    targets_list.append(targets)
                    lambdas_list.append(lambda_val)
                    lambda_val *= 0.7  # TD(λ) decay

        if inputs_list:
            self.inputs = np.stack(inputs_list)
            self.targets = np.stack(targets_list)
            self.lambdas = np.array(lambdas_list, dtype=np.float32)
        else:
            self.inputs = np.array([])
            self.targets = np.array([])
            self.lambdas = np.array([])

    def _build_role_batch(self, games: List[GameRecord]):
        """Build batch for role network training."""
        inputs_list = []
        targets_list = []

        for game in games:
            for decision in game.role_decisions:
                inputs_list.append(decision.inputs)

                # Target is softmax of action scores (scaled)
                scores = decision.action_scores
                # Use temperature=20 like the C code
                exp_scores = np.exp(20 * (scores / scores.max()))
                target = exp_scores / exp_scores.sum()
                targets_list.append(target)

        if inputs_list:
            self.inputs = np.stack(inputs_list)
            self.targets = np.stack(targets_list)
        else:
            self.inputs = np.array([])
            self.targets = np.array([])

    def to_torch(self):
        """Convert to PyTorch tensors."""
        import torch

        result = {
            'inputs': torch.from_numpy(self.inputs).float(),
            'targets': torch.from_numpy(self.targets).float(),
        }

        if self.network_type == 'eval' and hasattr(self, 'lambdas'):
            result['lambdas'] = torch.from_numpy(self.lambdas).float()

        return result


def merge_training_files(input_files: List[str], output_file: str):
    """Merge multiple training data files into one."""
    with TrainingDataWriter(output_file) as writer:
        for input_file in input_files:
            reader = TrainingDataReader(input_file)
            for game in reader:
                writer.write_game(game)


def get_training_stats(filepath: str) -> Dict[str, Any]:
    """Get statistics about a training data file."""
    reader = TrainingDataReader(filepath)

    stats = {
        'num_games': 0,
        'total_eval_states': 0,
        'total_role_decisions': 0,
        'by_expansion': {},
        'by_num_players': {},
    }

    for game in reader:
        stats['num_games'] += 1
        stats['total_eval_states'] += len(game.eval_states)
        stats['total_role_decisions'] += len(game.role_decisions)

        exp_key = str(game.expansion)
        if exp_key not in stats['by_expansion']:
            stats['by_expansion'][exp_key] = 0
        stats['by_expansion'][exp_key] += 1

        np_key = str(game.num_players)
        if np_key not in stats['by_num_players']:
            stats['by_num_players'][np_key] = 0
        stats['by_num_players'][np_key] += 1

    return stats


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # Print stats for a file
        filepath = sys.argv[1]
        stats = get_training_stats(filepath)
        print(f"Training data stats for {filepath}:")
        print(f"  Games: {stats['num_games']}")
        print(f"  Eval states: {stats['total_eval_states']}")
        print(f"  Role decisions: {stats['total_role_decisions']}")
        print(f"  By expansion: {stats['by_expansion']}")
        print(f"  By num_players: {stats['by_num_players']}")
    else:
        # Create example data
        print("Creating example training data...")

        # Create a fake game record
        game = GameRecord(
            game_id="test_123",
            expansion=0,
            num_players=2,
            advanced=False,
            random_seed=12345,
            winner_indices=[0],
            final_scores=[45, 38],
            num_rounds=10
        )

        # Add some fake eval states
        for i in range(5):
            state = EvalState(
                player_index=i % 2,
                round_num=i * 2,
                inputs=np.random.randn(704).astype(np.float32)
            )
            game.eval_states.append(state)

        # Add some fake role decisions
        for i in range(3):
            decision = RoleDecision(
                player_index=i % 2,
                round_num=i * 3,
                inputs=np.random.randn(605).astype(np.float32),
                chosen_action=2,
                action_scores=np.random.randn(7).astype(np.float32)
            )
            game.role_decisions.append(decision)

        # Write to file
        with TrainingDataWriter('/tmp/test_training.jsonl.gz') as writer:
            writer.write_game(game)

        # Read back
        reader = TrainingDataReader('/tmp/test_training.jsonl.gz')
        for loaded_game in reader:
            print(f"Loaded game: {loaded_game.game_id}")
            print(f"  Eval states: {len(loaded_game.eval_states)}")
            print(f"  Role decisions: {len(loaded_game.role_decisions)}")

        # Create batch
        batch = TrainingBatch([game], 'eval')
        print(f"Batch inputs shape: {batch.inputs.shape}")
        print(f"Batch targets shape: {batch.targets.shape}")
