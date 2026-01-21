"""
PyTorch implementation of the RFTG neural network.

This module provides a PyTorch implementation that matches the C neural network
in /home/rrenaud/rftg/net/src/net.c.

Network Architecture:
- Two-layer feedforward network
- Hidden layer: Linear + tanh activation
- Output layer: Linear + softmax activation

The C code uses an explicit bias node (input[num_inputs]=1.0, hidden[num_hidden]=1.0)
whereas PyTorch's Linear layer handles bias internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path


class RFTGNet(nn.Module):
    """
    PyTorch implementation of the RFTG neural network matching the C implementation.

    Attributes:
        num_inputs: Number of input features
        num_hidden: Number of hidden nodes
        num_outputs: Number of output classes
        input_names: List of input feature names (from .net file)
        num_training: Number of training iterations the network has seen
    """

    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        """
        Initialize the network with given dimensions.

        Args:
            num_inputs: Number of input features
            num_hidden: Number of hidden nodes
            num_outputs: Number of output classes
        """
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Create layers with bias
        self.hidden = nn.Linear(num_inputs, num_hidden)
        self.output = nn.Linear(num_hidden, num_outputs)

        # Metadata from .net file
        self.input_names: List[str] = []
        self.num_training: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        The C implementation does:
        1. hidden_sum[i] = sum(input[j] * hidden_weight[j][i]) for all j including bias
        2. hidden_result[i] = tanh(hidden_sum[i])
        3. output_sum[i] = sum(hidden_result[j] * output_weight[j][i]) for all j including bias
        4. Apply softmax with numerical stability (subtract max before exp)

        Args:
            x: Input tensor of shape (batch_size, num_inputs) or (num_inputs,)

        Returns:
            Output probabilities of shape (batch_size, num_outputs) or (num_outputs,)
        """
        # Handle both batched and unbatched input
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Hidden layer with tanh activation (matches C sigmoid() which is tanh)
        h = torch.tanh(self.hidden(x))

        # Output layer
        out = self.output(h)

        # Softmax for probabilities
        # The C code does: adj = -out[0], then exp(out[i] + adj) / sum
        # This is mathematically equivalent to softmax
        probs = F.softmax(out, dim=-1)

        if squeeze_output:
            probs = probs.squeeze(0)

        return probs

    def forward_with_intermediates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning intermediate values for debugging/testing.

        Args:
            x: Input tensor of shape (num_inputs,)

        Returns:
            Tuple of (hidden_result, output_result, win_prob)
            - hidden_result: Output of hidden layer after tanh, shape (num_hidden,)
            - output_result: Raw output sums before softmax, shape (num_outputs,)
            - win_prob: Final probabilities after softmax, shape (num_outputs,)
        """
        # Hidden layer
        hidden_sum = self.hidden(x)
        hidden_result = torch.tanh(hidden_sum)

        # Output layer
        output_result = self.output(hidden_result)

        # Apply softmax matching C implementation's numerical stability trick
        # C code: adj = -output[0], result[i] = exp(output[i] + adj)
        adj = -output_result[0]
        net_result = torch.exp(output_result + adj)
        prob_sum = net_result.sum()
        win_prob = net_result / prob_sum

        return hidden_result, output_result, win_prob


def load_net_file(filepath: str) -> RFTGNet:
    """
    Load a neural network from a .net file.

    File format (from net.c save_net/load_net):
    - Line 1: num_inputs num_hidden num_outputs
    - Line 2: num_training
    - Lines 3 to num_inputs+2: input names (one per line)
    - Next num_hidden * (num_inputs+1) lines: hidden weights
      - Organized as: for each hidden node i, for each input j (including bias at end)
        hidden_weight[j][i]
    - Next num_outputs * (num_hidden+1) lines: output weights
      - Organized as: for each output node i, for each hidden j (including bias at end)
        output_weight[j][i]

    Args:
        filepath: Path to the .net file

    Returns:
        Loaded RFTGNet model
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        # Read header
        header = f.readline().strip().split()
        num_inputs, num_hidden, num_outputs = int(header[0]), int(header[1]), int(header[2])

        # Read training count
        num_training = int(f.readline().strip())

        # Read input names
        input_names = []
        for _ in range(num_inputs):
            input_names.append(f.readline().rstrip('\n'))

        # Create network
        net = RFTGNet(num_inputs, num_hidden, num_outputs)
        net.input_names = input_names
        net.num_training = num_training

        # Read hidden weights
        # C format: for each hidden node i, for each input+bias j, hidden_weight[j][i]
        # hidden_weight is (num_inputs+1, num_hidden) in C
        # PyTorch Linear weight is (out_features, in_features) = (num_hidden, num_inputs)
        # PyTorch bias is (out_features,) = (num_hidden,)

        hidden_weights = []  # Will be (num_inputs+1, num_hidden)
        for i in range(num_hidden):
            col = []
            for j in range(num_inputs + 1):
                val = float(f.readline().strip())
                col.append(val)
            hidden_weights.append(col)

        # hidden_weights[i][j] = weight from input j to hidden i
        # PyTorch wants weight[i][j] = weight from input j to output i
        # So hidden_weights is already in the right shape conceptually
        # But we need to separate out the bias (last row in C is bias)

        # Convert to tensor: shape (num_hidden, num_inputs+1)
        hidden_tensor = torch.tensor(hidden_weights, dtype=torch.float64)

        # Extract weights and bias
        # In C: hidden_weight[j][i] where j=0..num_inputs-1 are inputs, j=num_inputs is bias
        # hidden_tensor[i][j] = hidden_weight[j][i]
        # So hidden_tensor[:, :-1] are the input weights, hidden_tensor[:, -1] is bias
        net.hidden.weight.data = hidden_tensor[:, :-1].float()
        net.hidden.bias.data = hidden_tensor[:, -1].float()

        # Read output weights
        # C format: for each output node i, for each hidden+bias j, output_weight[j][i]
        # output_weight is (num_hidden+1, num_output) in C

        output_weights = []  # Will be (num_outputs, num_hidden+1)
        for i in range(num_outputs):
            col = []
            for j in range(num_hidden + 1):
                val = float(f.readline().strip())
                col.append(val)
            output_weights.append(col)

        # Convert to tensor: shape (num_outputs, num_hidden+1)
        output_tensor = torch.tensor(output_weights, dtype=torch.float64)

        # Extract weights and bias
        net.output.weight.data = output_tensor[:, :-1].float()
        net.output.bias.data = output_tensor[:, -1].float()

    return net


def load_net_file_double(filepath: str) -> RFTGNet:
    """
    Load a neural network from a .net file using float64 (double precision).

    This matches the C implementation which uses double precision.
    Use this for regression testing to minimize numerical differences.

    Args:
        filepath: Path to the .net file

    Returns:
        Loaded RFTGNet model with float64 parameters
    """
    net = load_net_file(filepath)
    return net.double()


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rftg_net.py <network.net>")
        sys.exit(1)

    net_file = sys.argv[1]
    print(f"Loading network from {net_file}")

    net = load_net_file(net_file)
    print(f"Network dimensions: {net.num_inputs} -> {net.num_hidden} -> {net.num_outputs}")
    print(f"Training iterations: {net.num_training}")
    print(f"First 5 input names: {net.input_names[:5]}")

    # Test with random input
    x = torch.randn(net.num_inputs)
    probs = net(x)
    print(f"Output probabilities: {probs}")
    print(f"Sum of probabilities: {probs.sum():.10f}")
