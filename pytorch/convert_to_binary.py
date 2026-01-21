#!/usr/bin/env python3
"""
Convert RFTG training data from JSONL to compressed NPZ format.

This script converts existing training data batches from the gzipped JSONL format
to the more efficient NPZ format with bit-packing, achieving ~57x compression.

Usage:
    # Convert a single directory
    python convert_to_binary.py /path/to/training_data/batch_20240101_120000

    # Convert all batches in a directory
    python convert_to_binary.py /path/to/training_data --recursive

    # Convert and verify round-trip correctness
    python convert_to_binary.py /path/to/batch --verify

    # Dry run (show what would be converted)
    python convert_to_binary.py /path/to/training_data --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from training_data import (
    TrainingDataReader,
    BinaryBatchWriter,
    BinaryBatchReader,
    convert_jsonl_to_npz,
    GameRecord,
)


def find_jsonl_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find all JSONL files in a directory."""
    if recursive:
        return sorted(path.rglob("*.jsonl"))
    else:
        return sorted(path.glob("*.jsonl"))


def find_batch_dirs(path: Path) -> List[Path]:
    """Find all batch directories (containing worker_*.jsonl files)."""
    batch_dirs = []

    # Check if this directory itself contains JSONL files
    if list(path.glob("worker_*.jsonl")):
        batch_dirs.append(path)

    # Check subdirectories
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            if list(subdir.glob("worker_*.jsonl")):
                batch_dirs.append(subdir)

    return batch_dirs


def convert_batch_dir(batch_dir: Path, output_path: Path = None, verify: bool = False) -> Dict[str, Any]:
    """
    Convert all JSONL files in a batch directory to a single NPZ file.

    Args:
        batch_dir: Directory containing worker_*.jsonl files
        output_path: Output NPZ file path (default: batch_dir/batch.npz)
        verify: Whether to verify round-trip correctness

    Returns:
        Statistics about the conversion
    """
    if output_path is None:
        output_path = batch_dir / "batch.npz"

    # Find all JSONL files
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        return {'error': 'No JSONL files found'}

    # Create writer
    writer = BinaryBatchWriter(str(output_path))

    # Track statistics
    stats = {
        'batch_dir': str(batch_dir),
        'output_path': str(output_path),
        'num_files': len(jsonl_files),
        'num_games': 0,
        'num_eval': 0,
        'num_role': 0,
        'input_size': 0,
    }

    start_time = time.time()

    # Convert each JSONL file
    for jsonl_path in jsonl_files:
        stats['input_size'] += os.path.getsize(jsonl_path)

        reader = TrainingDataReader(str(jsonl_path))
        for game in reader:
            writer.add_game(game)
            stats['num_games'] += 1
            stats['num_eval'] += len(game.eval_states)
            stats['num_role'] += len(game.role_decisions)

    # Write output
    writer.write()

    stats['elapsed'] = time.time() - start_time
    stats['output_size'] = os.path.getsize(output_path)
    stats['compression_ratio'] = stats['input_size'] / stats['output_size'] if stats['output_size'] > 0 else 0

    # Verify if requested
    if verify:
        stats['verified'] = verify_conversion(jsonl_files, output_path)

    return stats


def verify_conversion(jsonl_files: List[Path], npz_path: Path) -> Dict[str, Any]:
    """
    Verify that NPZ file contains the same data as original JSONL files.

    Returns:
        Dict with verification results
    """
    # Load original data
    original_games = []
    for jsonl_path in jsonl_files:
        reader = TrainingDataReader(str(jsonl_path))
        original_games.extend(list(reader))

    # Load converted data
    reader = BinaryBatchReader(str(npz_path))
    converted_games = list(reader.iter_games())

    result = {
        'num_original_games': len(original_games),
        'num_converted_games': len(converted_games),
        'games_match': len(original_games) == len(converted_games),
    }

    if not result['games_match']:
        result['passed'] = False
        return result

    # Check each game
    mismatches = []
    for i, (orig, conv) in enumerate(zip(original_games, converted_games)):
        errors = []

        # Check game metadata
        if orig.game_id != conv.game_id:
            errors.append(f"game_id: {orig.game_id} vs {conv.game_id}")
        if orig.num_players != conv.num_players:
            errors.append(f"num_players: {orig.num_players} vs {conv.num_players}")
        if orig.winner_indices != conv.winner_indices:
            errors.append(f"winner_indices: {orig.winner_indices} vs {conv.winner_indices}")

        # Check eval states
        if len(orig.eval_states) != len(conv.eval_states):
            errors.append(f"eval_states count: {len(orig.eval_states)} vs {len(conv.eval_states)}")
        else:
            for j, (o_state, c_state) in enumerate(zip(orig.eval_states, conv.eval_states)):
                if o_state.player_index != c_state.player_index:
                    errors.append(f"eval[{j}].player_index")
                if o_state.round_num != c_state.round_num:
                    errors.append(f"eval[{j}].round_num")
                if not np.allclose(o_state.inputs, c_state.inputs, atol=1e-6):
                    errors.append(f"eval[{j}].inputs")

        # Check role decisions
        if len(orig.role_decisions) != len(conv.role_decisions):
            errors.append(f"role_decisions count: {len(orig.role_decisions)} vs {len(conv.role_decisions)}")
        else:
            for j, (o_dec, c_dec) in enumerate(zip(orig.role_decisions, conv.role_decisions)):
                if o_dec.player_index != c_dec.player_index:
                    errors.append(f"role[{j}].player_index")
                if o_dec.round_num != c_dec.round_num:
                    errors.append(f"role[{j}].round_num")
                if o_dec.chosen_action != c_dec.chosen_action:
                    errors.append(f"role[{j}].chosen_action")
                # Only check binary part of inputs (float part is stored separately)
                if not np.allclose(o_dec.inputs[:598], c_dec.inputs[:598], atol=1e-6):
                    errors.append(f"role[{j}].inputs (binary)")
                if not np.allclose(o_dec.action_scores, c_dec.action_scores, atol=1e-6):
                    errors.append(f"role[{j}].action_scores")

        if errors:
            mismatches.append({'game_index': i, 'errors': errors})

    result['num_mismatches'] = len(mismatches)
    result['passed'] = len(mismatches) == 0
    if mismatches:
        result['mismatches'] = mismatches[:5]  # Show first 5

    return result


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bytes"


def main():
    parser = argparse.ArgumentParser(
        description="Convert RFTG training data from JSONL to NPZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to batch directory or parent directory containing batches"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively find and convert all batch directories"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify round-trip correctness after conversion"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (for single batch conversion)"
    )
    parser.add_argument(
        "--delete-jsonl",
        action="store_true",
        help="Delete original JSONL files after successful conversion"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    # Find batch directories to convert
    if args.recursive:
        batch_dirs = find_batch_dirs(path)
    else:
        # Single directory
        batch_dirs = [path]

    if not batch_dirs:
        print("No batch directories found with JSONL files")
        sys.exit(1)

    print(f"Found {len(batch_dirs)} batch director{'ies' if len(batch_dirs) > 1 else 'y'} to convert")
    print()

    # Dry run - just show what would be converted
    if args.dry_run:
        total_input_size = 0
        for batch_dir in batch_dirs:
            jsonl_files = sorted(batch_dir.glob("*.jsonl"))
            batch_size = sum(os.path.getsize(f) for f in jsonl_files)
            total_input_size += batch_size
            print(f"  {batch_dir}: {len(jsonl_files)} files, {format_size(batch_size)}")
        print()
        print(f"Total: {format_size(total_input_size)}")
        print(f"Estimated output: {format_size(total_input_size // 57)} (57x compression)")
        sys.exit(0)

    # Convert each batch
    total_stats = {
        'num_batches': 0,
        'num_games': 0,
        'num_eval': 0,
        'num_role': 0,
        'input_size': 0,
        'output_size': 0,
        'elapsed': 0,
        'verified': 0,
        'failed': 0,
    }

    for batch_dir in batch_dirs:
        print(f"Converting: {batch_dir}")

        output_path = Path(args.output) if args.output and len(batch_dirs) == 1 else None
        stats = convert_batch_dir(batch_dir, output_path, verify=args.verify)

        if 'error' in stats:
            print(f"  Error: {stats['error']}")
            total_stats['failed'] += 1
            continue

        total_stats['num_batches'] += 1
        total_stats['num_games'] += stats['num_games']
        total_stats['num_eval'] += stats['num_eval']
        total_stats['num_role'] += stats['num_role']
        total_stats['input_size'] += stats['input_size']
        total_stats['output_size'] += stats['output_size']
        total_stats['elapsed'] += stats['elapsed']

        # Print stats
        print(f"  Games: {stats['num_games']:,}")
        print(f"  Samples: {stats['num_eval']:,} eval, {stats['num_role']:,} role")
        print(f"  Size: {format_size(stats['input_size'])} -> {format_size(stats['output_size'])} ({stats['compression_ratio']:.1f}x)")
        print(f"  Time: {stats['elapsed']:.1f}s")

        if args.verify:
            verification = stats.get('verified', {})
            if verification.get('passed'):
                print(f"  Verified: OK")
                total_stats['verified'] += 1
            else:
                print(f"  Verified: FAILED")
                if verification.get('mismatches'):
                    for m in verification['mismatches'][:3]:
                        print(f"    Game {m['game_index']}: {', '.join(m['errors'][:3])}")
                total_stats['failed'] += 1

        # Delete original files if requested and conversion succeeded
        if args.delete_jsonl and (not args.verify or stats.get('verified', {}).get('passed')):
            jsonl_files = sorted(batch_dir.glob("*.jsonl"))
            for f in jsonl_files:
                os.remove(f)
            print(f"  Deleted {len(jsonl_files)} JSONL files")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Batches converted: {total_stats['num_batches']}")
    print(f"Total games: {total_stats['num_games']:,}")
    print(f"Total samples: {total_stats['num_eval']:,} eval, {total_stats['num_role']:,} role")
    print(f"Size reduction: {format_size(total_stats['input_size'])} -> {format_size(total_stats['output_size'])}")
    if total_stats['output_size'] > 0:
        ratio = total_stats['input_size'] / total_stats['output_size']
        print(f"Compression ratio: {ratio:.1f}x")
    print(f"Total time: {total_stats['elapsed']:.1f}s")
    if args.verify:
        print(f"Verified: {total_stats['verified']} passed, {total_stats['failed']} failed")


if __name__ == "__main__":
    main()
