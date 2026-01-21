#!/usr/bin/env python3
"""
Regression tests comparing PyTorch implementation against C implementation.

This script loads test vectors exported from the C neural network and
compares them against the PyTorch implementation.
"""

import json
import argparse
import subprocess
import os
import sys
from pathlib import Path

import torch
import numpy as np

from rftg_net import RFTGNet, load_net_file, load_net_file_double


def generate_test_vectors(network_file: str, num_random: int = 10) -> dict:
    """
    Generate test vectors by running the C export tool.

    Args:
        network_file: Path to the .net file
        num_random: Number of random test cases to generate

    Returns:
        Dictionary containing test vectors
    """
    script_dir = Path(__file__).parent
    export_tool = script_dir / "export_test_vectors"

    if not export_tool.exists():
        raise FileNotFoundError(
            f"Export tool not found at {export_tool}. "
            "Please compile it first with: "
            "gcc -o export_test_vectors export_test_vectors.c "
            "-I../net/include -L../build/lib -lnet -lm"
        )

    result = subprocess.run(
        [str(export_tool), network_file, str(num_random)],
        capture_output=True,
        text=True,
        cwd=script_dir
    )

    if result.returncode != 0:
        raise RuntimeError(f"Export tool failed: {result.stderr}")

    return json.loads(result.stdout)


def compare_outputs(
    test_case: dict,
    pytorch_hidden: torch.Tensor,
    pytorch_output: torch.Tensor,
    pytorch_prob: torch.Tensor,
    tolerance: float = 1e-10
) -> tuple:
    """
    Compare C outputs with PyTorch outputs.

    Returns:
        Tuple of (passed, max_error, error_details)
    """
    c_hidden = torch.tensor(test_case["hidden_result"], dtype=torch.float64)
    c_output = torch.tensor(test_case["net_result"], dtype=torch.float64)
    c_prob = torch.tensor(test_case["win_prob"], dtype=torch.float64)

    hidden_diff = torch.abs(pytorch_hidden - c_hidden).max().item()
    prob_diff = torch.abs(pytorch_prob - c_prob).max().item()

    max_error = max(hidden_diff, prob_diff)
    passed = max_error < tolerance

    details = {
        "hidden_max_diff": hidden_diff,
        "prob_max_diff": prob_diff,
        "c_prob": c_prob.tolist(),
        "pytorch_prob": pytorch_prob.tolist(),
    }

    return passed, max_error, details


def run_regression_test(
    network_file: str,
    num_random: int = 10,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> bool:
    """
    Run regression tests for a single network file.

    Args:
        network_file: Path to the .net file
        num_random: Number of random test cases
        tolerance: Maximum acceptable error
        verbose: Print detailed output

    Returns:
        True if all tests pass
    """
    print(f"\nTesting network: {network_file}")
    print("=" * 60)

    # Generate test vectors from C implementation
    print("Generating test vectors from C implementation...")
    test_data = generate_test_vectors(network_file, num_random)

    # Load network in PyTorch (using double precision for accuracy)
    print("Loading network in PyTorch (float64)...")
    net = load_net_file_double(network_file)

    print(f"Network: {net.num_inputs} -> {net.num_hidden} -> {net.num_outputs}")
    print(f"Training iterations: {net.num_training}")

    # Run tests
    all_passed = True
    max_overall_error = 0.0
    failed_tests = []

    for test_case in test_data["test_cases"]:
        test_id = test_case["test_id"]

        # Create input tensor
        inputs = torch.tensor(test_case["inputs"], dtype=torch.float64)

        # Run PyTorch forward pass with intermediates
        with torch.no_grad():
            hidden, output, prob = net.forward_with_intermediates(inputs)

        # Compare outputs
        passed, max_error, details = compare_outputs(
            test_case, hidden, output, prob, tolerance
        )

        max_overall_error = max(max_overall_error, max_error)

        if not passed:
            all_passed = False
            failed_tests.append((test_id, max_error, details))

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  Test {test_id}: {status} (max_error={max_error:.2e})")

    # Print summary
    print(f"\nResults:")
    print(f"  Tests run: {len(test_data['test_cases'])}")
    print(f"  Tests passed: {len(test_data['test_cases']) - len(failed_tests)}")
    print(f"  Tests failed: {len(failed_tests)}")
    print(f"  Max error: {max_overall_error:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")

    if failed_tests:
        print(f"\nFailed tests:")
        for test_id, error, details in failed_tests[:5]:  # Show first 5 failures
            print(f"  Test {test_id}: max_error={error:.2e}")
            if verbose:
                print(f"    C prob: {details['c_prob']}")
                print(f"    PyTorch prob: {details['pytorch_prob']}")

    if all_passed:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")

    return all_passed


def run_all_networks_test(
    network_dir: str,
    network_type: str = "eval",
    num_random: int = 5,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> bool:
    """
    Run regression tests for all networks of a given type.

    Args:
        network_dir: Directory containing .net files
        network_type: "eval" or "role"
        num_random: Number of random test cases per network
        tolerance: Maximum acceptable error
        verbose: Print detailed output

    Returns:
        True if all tests pass
    """
    network_dir = Path(network_dir)
    pattern = f"rftg.{network_type}.*.net"

    network_files = sorted(network_dir.glob(pattern))
    if not network_files:
        print(f"No network files found matching {pattern} in {network_dir}")
        return False

    print(f"\nFound {len(network_files)} {network_type} networks to test")

    all_passed = True
    results = []

    for net_file in network_files:
        try:
            passed = run_regression_test(
                str(net_file), num_random, tolerance, verbose
            )
            results.append((net_file.name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"ERROR testing {net_file.name}: {e}")
            results.append((net_file.name, False))
            all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, p in results if p)
    print(f"Networks tested: {len(results)}")
    print(f"Networks passed: {passed_count}")
    print(f"Networks failed: {len(results) - passed_count}")

    if not all_passed:
        print("\nFailed networks:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Run regression tests comparing C and PyTorch neural network implementations"
    )
    parser.add_argument(
        "--network", "-n",
        help="Path to a specific .net file to test"
    )
    parser.add_argument(
        "--network-dir", "-d",
        default="../asset/network",
        help="Directory containing .net files (default: ../asset/network)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["eval", "role", "all"],
        default="eval",
        help="Type of networks to test (default: eval)"
    )
    parser.add_argument(
        "--num-random", "-r",
        type=int,
        default=10,
        help="Number of random test cases (default: 10)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Maximum acceptable error (default: 1e-5, typical max error ~1e-6)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Change to script directory for relative paths
    os.chdir(Path(__file__).parent)

    if args.network:
        # Test a specific network
        passed = run_regression_test(
            args.network,
            args.num_random,
            args.tolerance,
            args.verbose
        )
    else:
        # Test all networks
        if args.type == "all":
            passed_eval = run_all_networks_test(
                args.network_dir, "eval", args.num_random, args.tolerance, args.verbose
            )
            passed_role = run_all_networks_test(
                args.network_dir, "role", args.num_random, args.tolerance, args.verbose
            )
            passed = passed_eval and passed_role
        else:
            passed = run_all_networks_test(
                args.network_dir, args.type, args.num_random, args.tolerance, args.verbose
            )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
