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
from typing import List, Optional, Dict, Any, Tuple
import os
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
        # Handle missing game_id (C learner doesn't include it)
        game_id = d.get('game_id', f"{d.get('random_seed', 0)}")
        return cls(
            game_id=game_id,
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


# ============================================================================
# Binary Format (NPZ with bit-packing)
# ============================================================================

# Constants for input dimensions
EVAL_INPUT_DIM = 704  # All binary (-1/+1)
ROLE_INPUT_DIM = 605  # 598 binary + 7 floats (action scores)
ROLE_BINARY_DIM = 598  # First 598 are binary
ROLE_FLOAT_DIM = 7     # Last 7 are action scores (floats)


def pack_binary_inputs(inputs: np.ndarray) -> np.ndarray:
    """
    Pack binary -1/+1 inputs into bits.

    Args:
        inputs: Array of shape (N, D) with values -1 or +1

    Returns:
        Packed array of shape (N, ceil(D/8)) as uint8
    """
    # Convert -1/+1 to 0/1
    bits = (inputs == 1).astype(np.uint8)
    # Pack bits along axis 1
    return np.packbits(bits, axis=1)


def unpack_binary_inputs(packed: np.ndarray, original_dim: int) -> np.ndarray:
    """
    Unpack bit-packed data back to -1/+1 format.

    Args:
        packed: Packed array of shape (N, ceil(D/8)) as uint8
        original_dim: Original dimension before packing

    Returns:
        Array of shape (N, original_dim) with values -1 or +1
    """
    # Unpack bits
    unpacked = np.unpackbits(packed, axis=1)
    # Trim to original dimension (unpackbits pads to multiple of 8)
    unpacked = unpacked[:, :original_dim]
    # Convert 0/1 back to -1/+1
    return unpacked.astype(np.float32) * 2 - 1


class BinaryBatchWriter:
    """
    Writes training data in compressed binary NPZ format.

    Format achieves ~57x compression vs gzipped JSONL by:
    - Bit-packing binary -1/+1 inputs (704 bits → 88 bytes per sample)
    - Using appropriate dtypes for metadata
    - Leveraging numpy's built-in compression

    File structure:
        eval_packed: bit-packed eval inputs, shape (N_eval, 88)
        eval_meta: [player_idx, round_num], shape (N_eval, 2), int8
        role_packed: bit-packed role binary inputs, shape (N_role, 75)
        role_floats: action scores, shape (N_role, 7), float32
        role_meta: [player_idx, round_num, chosen_action], shape (N_role, 3), int16
        game_boundaries: [eval_start, role_start], shape (N_games, 2), int32
        game_info: structured array with game metadata
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Accumulators for batch data
        self.eval_inputs = []
        self.eval_meta = []  # (player_idx, round_num)
        self.role_inputs = []
        self.role_floats = []
        self.role_meta = []  # (player_idx, round_num, chosen_action)

        # Game boundary tracking
        self.game_boundaries = []  # (eval_start_idx, role_start_idx)
        self.game_info = []  # List of game metadata dicts

    def add_game(self, game: GameRecord):
        """Add a single game to the batch."""
        # Record game boundary
        eval_start = len(self.eval_inputs)
        role_start = len(self.role_inputs)
        self.game_boundaries.append((eval_start, role_start))

        # Store game info
        self.game_info.append({
            'game_id': game.game_id,
            'random_seed': game.random_seed,
            'expansion': game.expansion,
            'num_players': game.num_players,
            'advanced': game.advanced,
            'winner_indices': game.winner_indices,
            'final_scores': game.final_scores,
            'num_rounds': game.num_rounds,
        })

        # Add eval states
        for state in game.eval_states:
            self.eval_inputs.append(state.inputs)
            self.eval_meta.append([state.player_index, state.round_num])

        # Add role decisions
        for decision in game.role_decisions:
            # Split role inputs into binary part and float part
            binary_part = decision.inputs[:ROLE_BINARY_DIM]
            float_part = decision.action_scores
            self.role_inputs.append(binary_part)
            self.role_floats.append(float_part)
            self.role_meta.append([
                decision.player_index,
                decision.round_num,
                decision.chosen_action
            ])

    def write(self):
        """Write all accumulated data to the NPZ file."""
        if not self.eval_inputs and not self.role_inputs:
            return

        # Convert lists to arrays
        eval_inputs = np.array(self.eval_inputs, dtype=np.float32) if self.eval_inputs else np.array([], dtype=np.float32).reshape(0, EVAL_INPUT_DIM)
        eval_meta = np.array(self.eval_meta, dtype=np.int8) if self.eval_meta else np.array([], dtype=np.int8).reshape(0, 2)

        role_inputs = np.array(self.role_inputs, dtype=np.float32) if self.role_inputs else np.array([], dtype=np.float32).reshape(0, ROLE_BINARY_DIM)
        role_floats = np.array(self.role_floats, dtype=np.float32) if self.role_floats else np.array([], dtype=np.float32).reshape(0, ROLE_FLOAT_DIM)
        role_meta = np.array(self.role_meta, dtype=np.int16) if self.role_meta else np.array([], dtype=np.int16).reshape(0, 3)

        game_boundaries = np.array(self.game_boundaries, dtype=np.int32) if self.game_boundaries else np.array([], dtype=np.int32).reshape(0, 2)

        # Pack binary inputs
        eval_packed = pack_binary_inputs(eval_inputs) if len(eval_inputs) > 0 else np.array([], dtype=np.uint8).reshape(0, (EVAL_INPUT_DIM + 7) // 8)
        role_packed = pack_binary_inputs(role_inputs) if len(role_inputs) > 0 else np.array([], dtype=np.uint8).reshape(0, (ROLE_BINARY_DIM + 7) // 8)

        # Serialize game_info as JSON bytes
        game_info_json = json.dumps(self.game_info).encode('utf-8')
        game_info_bytes = np.frombuffer(game_info_json, dtype=np.uint8)

        # Save compressed NPZ
        np.savez_compressed(
            self.filepath,
            eval_packed=eval_packed,
            eval_meta=eval_meta,
            role_packed=role_packed,
            role_floats=role_floats,
            role_meta=role_meta,
            game_boundaries=game_boundaries,
            game_info_bytes=game_info_bytes,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()


class BinaryBatchReader:
    """
    Reads training data from compressed binary NPZ format.

    Provides efficient access to training data without parsing JSON.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._data = None

    def _load(self):
        """Load data lazily."""
        if self._data is None:
            self._data = np.load(self.filepath, allow_pickle=False)

    @property
    def num_eval_samples(self) -> int:
        """Number of eval training samples."""
        self._load()
        return len(self._data['eval_packed'])

    @property
    def num_role_samples(self) -> int:
        """Number of role training samples."""
        self._load()
        return len(self._data['role_packed'])

    @property
    def num_games(self) -> int:
        """Number of games in this batch."""
        self._load()
        return len(self._data['game_boundaries'])

    def get_eval_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all eval training data.

        Returns:
            inputs: Shape (N, 704), float32 with values -1/+1
            meta: Shape (N, 2), int8 with [player_idx, round_num]
        """
        self._load()
        inputs = unpack_binary_inputs(self._data['eval_packed'], EVAL_INPUT_DIM)
        meta = self._data['eval_meta']
        return inputs, meta

    def get_role_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all role training data.

        Returns:
            inputs: Shape (N, 598), float32 with values -1/+1 (binary part)
            floats: Shape (N, 7), float32 action scores
            meta: Shape (N, 3), int16 with [player_idx, round_num, chosen_action]
        """
        self._load()
        inputs = unpack_binary_inputs(self._data['role_packed'], ROLE_BINARY_DIM)
        floats = self._data['role_floats']
        meta = self._data['role_meta']
        return inputs, floats, meta

    def get_full_role_inputs(self) -> np.ndarray:
        """
        Get role inputs with binary and float parts concatenated.

        Returns:
            inputs: Shape (N, 605), float32
        """
        self._load()
        binary = unpack_binary_inputs(self._data['role_packed'], ROLE_BINARY_DIM)
        floats = self._data['role_floats']
        return np.concatenate([binary, floats], axis=1)

    def get_game_info(self) -> List[Dict]:
        """Get metadata for all games in this batch."""
        self._load()
        game_info_bytes = self._data['game_info_bytes'].tobytes()
        return json.loads(game_info_bytes.decode('utf-8'))

    def get_game_boundaries(self) -> np.ndarray:
        """
        Get game boundary indices.

        Returns:
            Array of shape (N_games, 2) with [eval_start, role_start] for each game
        """
        self._load()
        return self._data['game_boundaries']

    def iter_games(self):
        """
        Iterate over games, yielding reconstructed GameRecord objects.

        Note: This is slower than accessing batch data directly.
        Use get_eval_data() and get_role_data() for training.
        """
        self._load()
        game_info = self.get_game_info()
        boundaries = self.get_game_boundaries()
        eval_inputs, eval_meta = self.get_eval_data()
        role_inputs, role_floats, role_meta = self.get_role_data()

        for i, info in enumerate(game_info):
            eval_start = boundaries[i, 0]
            role_start = boundaries[i, 1]
            eval_end = boundaries[i + 1, 0] if i + 1 < len(boundaries) else len(eval_inputs)
            role_end = boundaries[i + 1, 1] if i + 1 < len(boundaries) else len(role_inputs)

            # Reconstruct eval states
            eval_states = []
            for j in range(eval_start, eval_end):
                eval_states.append(EvalState(
                    player_index=int(eval_meta[j, 0]),
                    round_num=int(eval_meta[j, 1]),
                    inputs=eval_inputs[j],
                ))

            # Reconstruct role decisions
            role_decisions = []
            for j in range(role_start, role_end):
                # Reconstruct full input by concatenating binary and float parts
                full_input = np.concatenate([role_inputs[j], role_floats[j]])
                role_decisions.append(RoleDecision(
                    player_index=int(role_meta[j, 0]),
                    round_num=int(role_meta[j, 1]),
                    inputs=full_input,
                    chosen_action=int(role_meta[j, 2]),
                    action_scores=role_floats[j],
                ))

            yield GameRecord(
                game_id=info['game_id'],
                expansion=info['expansion'],
                num_players=info['num_players'],
                advanced=info['advanced'],
                random_seed=info['random_seed'],
                winner_indices=info['winner_indices'],
                final_scores=info['final_scores'],
                eval_states=eval_states,
                role_decisions=role_decisions,
                num_rounds=info['num_rounds'],
            )

    def close(self):
        """Close the file handle."""
        if self._data is not None:
            self._data.close()
            self._data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def convert_jsonl_to_npz(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convert a gzipped JSONL file to compressed NPZ format.

    Args:
        input_path: Path to input .jsonl.gz file
        output_path: Path to output .npz file

    Returns:
        Statistics about the conversion
    """
    reader = TrainingDataReader(input_path)
    writer = BinaryBatchWriter(output_path)

    stats = {'num_games': 0, 'num_eval': 0, 'num_role': 0}

    for game in reader:
        writer.add_game(game)
        stats['num_games'] += 1
        stats['num_eval'] += len(game.eval_states)
        stats['num_role'] += len(game.role_decisions)

    writer.write()

    # Get file sizes
    stats['input_size'] = os.path.getsize(input_path)
    stats['output_size'] = os.path.getsize(output_path)
    stats['compression_ratio'] = stats['input_size'] / stats['output_size'] if stats['output_size'] > 0 else 0

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
