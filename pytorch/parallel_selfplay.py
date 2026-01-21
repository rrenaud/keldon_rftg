#!/usr/bin/env python3
"""
Parallel self-play runner for RFTG training.

Runs multiple instances of the learner in parallel to generate games faster.
Each worker uses a different random seed and can write to separate output files.

Usage:
    python parallel_selfplay.py --workers 4 --games-per-worker 100 -p 2 -e 0

For now, this runs the standard learner which does online training.
Once we add data export to the C code, this will capture training data.
"""

import argparse
import subprocess
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import random

from training_data import GameRecord, BinaryBatchWriter
import json


def convert_batch_to_binary(batch_dir: str, delete_jsonl: bool = True) -> Dict[str, Any]:
    """
    Convert a batch directory from JSONL to NPZ format.

    Args:
        batch_dir: Directory containing worker_*.jsonl files
        delete_jsonl: Whether to delete JSONL files after conversion

    Returns:
        Conversion statistics
    """
    batch_path = Path(batch_dir)
    jsonl_files = sorted(batch_path.glob("*.jsonl"))

    if not jsonl_files:
        return {'error': 'No JSONL files found'}

    output_path = batch_path / "batch.npz"
    writer = BinaryBatchWriter(str(output_path))

    stats = {
        'num_files': len(jsonl_files),
        'num_games': 0,
        'input_size': 0,
    }

    for jsonl_path in jsonl_files:
        stats['input_size'] += os.path.getsize(jsonl_path)
        # Read plain JSONL files (not gzipped) from the C learner
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    game = GameRecord.from_dict(json.loads(line))
                    writer.add_game(game)
                    stats['num_games'] += 1

    writer.write()

    stats['output_size'] = os.path.getsize(output_path)
    stats['compression_ratio'] = stats['input_size'] / stats['output_size'] if stats['output_size'] > 0 else 0

    # Delete JSONL files if requested
    if delete_jsonl:
        for f in jsonl_files:
            os.remove(f)
        stats['deleted_files'] = len(jsonl_files)

    return stats


def find_learner_binary() -> Optional[Path]:
    """Find the learner binary."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent / "build" / "bin" / "learner",
        Path(__file__).parent.parent / "build" / "learner",
        Path("/home/rrenaud/rftg/build/bin/learner"),
    ]

    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path

    return None


def run_worker(
    worker_id: int,
    learner_path: str,
    num_games: int,
    num_players: int,
    expansion: int,
    advanced: bool,
    random_seed: int,
    working_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single worker process.

    Returns:
        Dict with worker results including output_file path if export was enabled
    """
    start_time = time.time()

    # Build command
    cmd = [
        learner_path,
        "-n", str(num_games),
        "-p", str(num_players),
        "-e", str(expansion),
        "-r", str(random_seed),
        "-f", "0.0",  # No training, just play (factor=0 means no weight updates)
    ]

    if advanced:
        cmd.append("-a")

    if verbose:
        cmd.append("-v")

    # Add export file if output directory specified
    output_file = None
    if output_dir:
        output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        cmd.extend(["-x", output_file])

    # Run learner from the asset directory (where cards.txt lives)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=working_dir
        )

        elapsed = time.time() - start_time
        games_per_sec = num_games / elapsed if elapsed > 0 else 0

        # Parse output for scores
        scores = []
        for line in result.stdout.split('\n'):
            if line.startswith('Player ') and ':' in line:
                try:
                    score = int(line.split(':')[1].strip())
                    scores.append(score)
                except:
                    pass

        return {
            'worker_id': worker_id,
            'success': result.returncode == 0,
            'num_games': num_games,
            'elapsed': elapsed,
            'games_per_sec': games_per_sec,
            'scores': scores,
            'stderr': result.stderr if result.returncode != 0 else '',
            'output_file': output_file,
        }

    except subprocess.TimeoutExpired:
        return {
            'worker_id': worker_id,
            'success': False,
            'error': 'timeout',
            'num_games': num_games,
        }
    except Exception as e:
        return {
            'worker_id': worker_id,
            'success': False,
            'error': str(e),
            'num_games': num_games,
        }


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


def run_parallel_selfplay(
    num_workers: int,
    games_per_worker: int,
    num_players: int = 2,
    expansion: int = 0,
    advanced: bool = False,
    base_seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    output_format: str = 'jsonl',
) -> Dict[str, Any]:
    """
    Run self-play in parallel across multiple workers.

    Args:
        num_workers: Number of parallel worker processes
        games_per_worker: Number of games each worker plays
        num_players: Number of players per game
        expansion: Expansion level (0-4)
        advanced: Use advanced 2-player rules
        base_seed: Base random seed (workers use base_seed + worker_id)
        output_dir: Directory to save output (optional)
        verbose: Print verbose output
        output_format: Output format ('jsonl' or 'binary')

    Returns:
        Dict with aggregate results
    """
    learner_path = find_learner_binary()
    if not learner_path:
        raise FileNotFoundError(
            "Could not find learner binary. "
            "Please build the project first with: "
            "cd /home/rrenaud/rftg/build && cmake .. && make"
        )

    asset_dir = find_asset_dir()
    if not asset_dir:
        raise FileNotFoundError(
            "Could not find asset directory (cards.txt). "
            "Expected at /home/rrenaud/rftg/asset/"
        )

    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Starting {num_workers} workers, {games_per_worker} games each")
    print(f"Total games: {num_workers * games_per_worker}")
    print(f"Configuration: {num_players} players, expansion {expansion}, "
          f"{'advanced' if advanced else 'standard'}")
    print(f"Learner: {learner_path}")
    print(f"Asset dir: {asset_dir}")
    print(f"Base seed: {base_seed}")
    if output_dir:
        print(f"Output dir: {output_dir}")
    print()

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id in range(num_workers):
            seed = base_seed + worker_id * 1000000  # Ensure non-overlapping seeds
            future = executor.submit(
                run_worker,
                worker_id,
                str(learner_path),
                games_per_worker,
                num_players,
                expansion,
                advanced,
                seed,
                str(asset_dir),  # Working directory
                output_dir,
                verbose
            )
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['success']:
                print(f"Worker {result['worker_id']}: completed {result['num_games']} games "
                      f"in {result['elapsed']:.1f}s ({result['games_per_sec']:.1f} games/sec)")
            else:
                print(f"Worker {result['worker_id']}: FAILED - {result.get('error', 'unknown error')}")

    total_elapsed = time.time() - start_time
    successful = [r for r in results if r['success']]
    total_games = sum(r['num_games'] for r in successful)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Workers: {len(successful)}/{num_workers} successful")
    print(f"Total games: {total_games}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Overall rate: {total_games / total_elapsed:.1f} games/sec")

    # Report on output files
    output_files = [r.get('output_file') for r in successful if r.get('output_file')]
    if output_files:
        total_size = sum(os.path.getsize(f) for f in output_files if os.path.exists(f))
        print(f"Output files: {len(output_files)} files, {total_size / 1024 / 1024:.1f} MB total")
        print(f"Output directory: {output_dir}")

    # Convert to binary format if requested
    npz_file = None
    if output_format == 'binary' and output_dir and output_files:
        print("Converting to binary format...")
        conv_stats = convert_batch_to_binary(output_dir, delete_jsonl=True)
        if 'error' not in conv_stats:
            npz_file = os.path.join(output_dir, "batch.npz")
            print(f"Converted: {conv_stats['input_size'] / 1024 / 1024:.1f} MB -> "
                  f"{conv_stats['output_size'] / 1024 / 1024:.1f} MB "
                  f"({conv_stats['compression_ratio']:.1f}x compression)")
        else:
            print(f"Conversion failed: {conv_stats['error']}")

    return {
        'num_workers': num_workers,
        'successful_workers': len(successful),
        'total_games': total_games,
        'total_elapsed': total_elapsed,
        'games_per_sec': total_games / total_elapsed,
        'worker_results': results,
        'output_files': output_files if output_format == 'jsonl' else ([npz_file] if npz_file else []),
    }


def run_continuous_selfplay(
    num_workers: int,
    games_per_batch: int,
    num_players: int = 2,
    expansion: int = 0,
    advanced: bool = False,
    output_dir: str = "./training_data",
    verbose: bool = False,
    output_format: str = 'jsonl',
):
    """
    Run self-play continuously until interrupted.

    Each batch creates timestamped output files to avoid overwriting.
    Press Ctrl+C to stop gracefully.
    """
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    learner_path = find_learner_binary()
    if not learner_path:
        raise FileNotFoundError("Could not find learner binary.")

    asset_dir = find_asset_dir()
    if not asset_dir:
        raise FileNotFoundError("Could not find asset directory.")

    print("=" * 60)
    print("CONTINUOUS SELF-PLAY MODE")
    print("=" * 60)
    print(f"Workers: {num_workers}")
    print(f"Games per batch: {games_per_batch} ({games_per_batch // num_workers} per worker)")
    print(f"Configuration: {num_players} players, expansion {expansion}")
    print(f"Output directory: {output_dir}")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 60)
    print()

    total_games = 0
    total_time = 0
    batch_num = 0
    start_time = time.time()

    try:
        while True:
            batch_num += 1
            batch_start = time.time()

            # Create timestamped subdirectory for this batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
            os.makedirs(batch_dir, exist_ok=True)

            # Generate random seed for this batch
            base_seed = random.randint(0, 2**31 - 1)

            print(f"[Batch {batch_num}] Starting {games_per_batch} games...")

            # Run workers
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                games_per_worker = games_per_batch // num_workers

                for worker_id in range(num_workers):
                    seed = base_seed + worker_id * 1000000
                    future = executor.submit(
                        run_worker,
                        worker_id,
                        str(learner_path),
                        games_per_worker,
                        num_players,
                        expansion,
                        advanced,
                        seed,
                        str(asset_dir),
                        batch_dir,
                        verbose
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    results.append(future.result())

            # Summarize batch
            batch_elapsed = time.time() - batch_start
            successful = [r for r in results if r['success']]
            batch_games = sum(r['num_games'] for r in successful)
            total_games += batch_games
            total_time = time.time() - start_time

            # Convert to binary format if requested
            if output_format == 'binary':
                conv_stats = convert_batch_to_binary(batch_dir, delete_jsonl=True)
                batch_size = conv_stats.get('output_size', 0)
                file_ext = '.npz'
            else:
                batch_size = sum(
                    os.path.getsize(os.path.join(batch_dir, f))
                    for f in os.listdir(batch_dir) if f.endswith('.jsonl')
                )
                file_ext = '.jsonl'

            # Get total size
            total_size = 0
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    if f.endswith(file_ext):
                        total_size += os.path.getsize(os.path.join(root, f))

            print(f"[Batch {batch_num}] Completed: {batch_games} games in {batch_elapsed:.1f}s "
                  f"({batch_games/batch_elapsed:.1f} g/s) | "
                  f"Batch: {batch_size/1024/1024:.1f}MB | "
                  f"Total: {total_games:,} games, {total_size/1024/1024:.1f}MB, "
                  f"{total_games/total_time:.1f} g/s avg")

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("STOPPED BY USER")
        print("=" * 60)
        print(f"Total batches: {batch_num}")
        print(f"Total games: {total_games:,}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average rate: {total_games/total_time:.1f} games/sec")
        print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel self-play for RFTG training"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Number of parallel workers (default: {os.cpu_count() or 4})"
    )
    parser.add_argument(
        "--games-per-worker", "-n",
        type=int,
        default=100,
        help="Number of games per worker (default: 100)"
    )
    parser.add_argument(
        "--players", "-p",
        type=int,
        default=2,
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--expansion", "-e",
        type=int,
        default=0,
        help="Expansion level 0-4 (default: 0 = base game)"
    )
    parser.add_argument(
        "--advanced", "-a",
        action="store_true",
        help="Use advanced 2-player rules"
    )
    parser.add_argument(
        "--seed", "-r",
        type=int,
        default=None,
        help="Base random seed (default: random)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for training data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously until Ctrl+C (uses --games-per-worker as batch size)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["jsonl", "binary"],
        default="jsonl",
        help="Output format: 'jsonl' (legacy) or 'binary' (NPZ, ~57x smaller)"
    )

    args = parser.parse_args()

    try:
        if args.continuous:
            if not args.output_dir:
                args.output_dir = "./training_data"
            run_continuous_selfplay(
                num_workers=args.workers,
                games_per_batch=args.workers * args.games_per_worker,
                num_players=args.players,
                expansion=args.expansion,
                advanced=args.advanced,
                output_dir=args.output_dir,
                verbose=args.verbose,
                output_format=args.format,
            )
        else:
            results = run_parallel_selfplay(
                num_workers=args.workers,
                games_per_worker=args.games_per_worker,
                num_players=args.players,
                expansion=args.expansion,
                advanced=args.advanced,
                base_seed=args.seed,
                output_dir=args.output_dir,
                verbose=args.verbose,
                output_format=args.format,
            )
            sys.exit(0 if results['successful_workers'] == results['num_workers'] else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
