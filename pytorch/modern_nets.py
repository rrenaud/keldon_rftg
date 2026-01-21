"""
Modern neural network architectures for RFTG eval network.

This module provides more sophisticated architectures beyond the baseline
2-layer MLP, including residual networks with modern techniques like
LayerNorm, GELU activations, and dropout.

Also includes:
- Card embedding modules for learned card representations
- Attention-based aggregation for variable-length card sets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block with GELU activation.

    Architecture:
        LayerNorm → Linear(expansion_factor * hidden_dim) → GELU →
        Linear(hidden_dim) → Dropout → Add residual

    Uses pre-normalization (LayerNorm before transformations) which has been
    shown to improve training stability in deep networks.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the residual block.

        Args:
            hidden_dim: Dimension of input and output
            expansion_factor: Factor to expand hidden dimension in MLP
            dropout: Dropout probability
        """
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class ResidualMLP(nn.Module):
    """
    Modern residual MLP for RFTG eval network.

    Architecture:
        Input(num_inputs) → Linear(hidden_dim)
        → [ResidualBlock × num_layers]
        → LayerNorm → Linear(num_outputs) → Softmax

    Each ResidualBlock:
        LayerNorm → Linear(4x) → GELU → Linear(1x) → Dropout → Add residual

    This architecture provides:
    - Better gradient flow through residual connections
    - Training stability via LayerNorm
    - Smoother activations via GELU
    - Regularization via Dropout
    """

    # Predefined configurations
    CONFIGS = {
        'small': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1},
        'medium': {'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.1},
        'large': {'hidden_dim': 512, 'num_layers': 6, 'dropout': 0.15},
    }

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the ResidualMLP.

        Args:
            num_inputs: Number of input features (704 for RFTG eval)
            num_outputs: Number of output classes (2 for win probability)
            hidden_dim: Hidden dimension size
            num_layers: Number of residual blocks
            expansion_factor: MLP expansion factor in residual blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(num_inputs, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expansion_factor, dropout)
            for _ in range(num_layers)
        ])

        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_outputs)

        # For compatibility with baseline RFTGNet interface
        self.num_hidden = hidden_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_config(
        cls,
        config_name: str,
        num_inputs: int,
        num_outputs: int,
    ) -> 'ResidualMLP':
        """
        Create a ResidualMLP from a predefined configuration.

        Args:
            config_name: One of 'small', 'medium', 'large'
            num_inputs: Number of input features
            num_outputs: Number of output classes

        Returns:
            Configured ResidualMLP instance
        """
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. "
                           f"Available: {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[config_name]
        return cls(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

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

        # Input projection
        x = self.input_proj(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Softmax for probabilities
        probs = F.softmax(x, dim=-1)

        if squeeze_output:
            probs = probs.squeeze(0)

        return probs

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (before softmax).

        Useful for training with CrossEntropyLoss which applies its own softmax.

        Args:
            x: Input tensor of shape (batch_size, num_inputs) or (num_inputs,)

        Returns:
            Raw logits of shape (batch_size, num_outputs) or (num_outputs,)
        """
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        logits = self.output_proj(x)

        if squeeze_output:
            logits = logits.squeeze(0)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'expansion_factor': self.blocks[0].fc1.out_features // self.hidden_dim,
            'dropout': self.blocks[0].dropout.p,
        }


class UnifiedRFTGNet(nn.Module):
    """
    Unified network with shared backbone and separate value/policy heads.

    This architecture follows AlphaZero-style design:
    - Shared residual backbone learns common representations
    - Value head predicts win probability (like eval network)
    - Policy head predicts action distribution (like role network)

    Benefits:
    - Parameter sharing improves sample efficiency
    - Joint training can improve both heads
    - Single forward pass for both outputs during gameplay
    """

    CONFIGS = {
        'small': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1},
        'medium': {'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.1},
        'large': {'hidden_dim': 512, 'num_layers': 6, 'dropout': 0.15},
    }

    def __init__(
        self,
        num_inputs: int,
        num_value_outputs: int = 2,
        num_policy_outputs: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize the unified network.

        Args:
            num_inputs: Number of input features (704 for 2p base game)
            num_value_outputs: Number of value outputs (2 for 2p win prob)
            num_policy_outputs: Number of policy outputs (7 for action selection)
            hidden_dim: Hidden dimension size in backbone
            num_layers: Number of residual blocks
            expansion_factor: MLP expansion factor in residual blocks
            dropout: Dropout probability
            head_hidden_dim: Hidden dim for output heads (defaults to hidden_dim)
        """
        super().__init__()

        self.num_inputs = num_inputs
        self.num_value_outputs = num_value_outputs
        self.num_policy_outputs = num_policy_outputs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if head_hidden_dim is None:
            head_hidden_dim = hidden_dim

        # Shared backbone
        self.input_proj = nn.Linear(num_inputs, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expansion_factor, dropout)
            for _ in range(num_layers)
        ])

        # Value head (win probability estimation)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_value_outputs),
        )

        # Policy head (action selection)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_policy_outputs),
        )

        # For compatibility
        self.num_hidden = hidden_dim
        self.num_outputs = num_value_outputs  # For backward compatibility

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_config(
        cls,
        config_name: str,
        num_inputs: int,
        num_value_outputs: int = 2,
        num_policy_outputs: int = 7,
    ) -> 'UnifiedRFTGNet':
        """
        Create a UnifiedRFTGNet from a predefined configuration.

        Args:
            config_name: One of 'small', 'medium', 'large'
            num_inputs: Number of input features
            num_value_outputs: Number of value outputs
            num_policy_outputs: Number of policy outputs

        Returns:
            Configured UnifiedRFTGNet instance
        """
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. "
                           f"Available: {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[config_name]
        return cls(
            num_inputs=num_inputs,
            num_value_outputs=num_value_outputs,
            num_policy_outputs=num_policy_outputs,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared backbone only.

        Args:
            x: Input tensor of shape (batch_size, num_inputs)

        Returns:
            Backbone features of shape (batch_size, hidden_dim)
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both value and policy outputs.

        Args:
            x: Input tensor of shape (batch_size, num_inputs) or (num_inputs,)
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            Tuple of (value, policy):
                - value: Win probabilities (batch_size, num_value_outputs)
                - policy: Action probabilities (batch_size, num_policy_outputs)
        """
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Shared backbone
        features = self.forward_backbone(x)

        # Separate heads
        value_logits = self.value_head(features)
        policy_logits = self.policy_head(features)

        if return_logits:
            value = value_logits
            policy = policy_logits
        else:
            value = F.softmax(value_logits, dim=-1)
            policy = F.softmax(policy_logits, dim=-1)

        if squeeze_output:
            value = value.squeeze(0)
            policy = policy.squeeze(0)

        return value, policy

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only value output (for compatibility).

        Args:
            x: Input tensor

        Returns:
            Win probabilities
        """
        value, _ = self.forward(x)
        return value

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only policy output.

        Args:
            x: Input tensor

        Returns:
            Action probabilities
        """
        _, policy = self.forward(x)
        return policy

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'num_inputs': self.num_inputs,
            'num_value_outputs': self.num_value_outputs,
            'num_policy_outputs': self.num_policy_outputs,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'expansion_factor': self.blocks[0].fc1.out_features // self.hidden_dim,
            'dropout': self.blocks[0].dropout.p,
        }


class CardEmbedding(nn.Module):
    """
    Learned embeddings for RFTG cards.

    Each card gets a learned embedding vector that captures its properties
    and strategic value. Similar cards (e.g., military worlds) will learn
    similar representations.
    """

    def __init__(
        self,
        num_cards: int = 200,
        embed_dim: int = 32,
        include_special: bool = True,
    ):
        """
        Initialize card embeddings.

        Args:
            num_cards: Total number of unique cards
            embed_dim: Dimension of each card embedding
            include_special: Whether to include special tokens (pad, unknown)
        """
        super().__init__()

        self.num_cards = num_cards
        self.embed_dim = embed_dim
        self.include_special = include_special

        # Add 2 for PAD (0) and UNKNOWN (1) tokens if special tokens enabled
        vocab_size = num_cards + 2 if include_special else num_cards
        self.pad_idx = 0 if include_special else None
        self.unk_idx = 1 if include_special else None

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.pad_idx,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small random values."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.pad_idx])

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for card IDs.

        Args:
            card_ids: Tensor of card IDs, shape (batch, num_cards) or (num_cards,)
                     Card IDs should be offset by 2 if include_special=True

        Returns:
            Card embeddings, shape (..., embed_dim)
        """
        return self.embedding(card_ids)


class CardSetEncoder(nn.Module):
    """
    Encodes a variable-length set of cards into a fixed-size representation
    using multi-head attention.

    This allows the network to learn card synergies and interactions
    rather than treating each card position independently.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        """
        Initialize the card set encoder.

        Args:
            embed_dim: Dimension of card embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_cls_token: Whether to use a CLS token for aggregation
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token

        # CLS token for aggregation
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        card_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a set of card embeddings into a single vector.

        Args:
            card_embeddings: Shape (batch, num_cards, embed_dim)
            mask: Optional boolean mask, True for valid positions
                  Shape (batch, num_cards)

        Returns:
            Aggregated representation, shape (batch, embed_dim)
        """
        batch_size = card_embeddings.size(0)

        if self.use_cls_token:
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, card_embeddings], dim=1)

            # Extend mask for CLS token
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        else:
            x = card_embeddings

        # Convert mask to attention mask format (True = ignore)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True means ignore

        # Self-attention with residual
        attended, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attended)

        # Feedforward with residual
        x = self.norm2(x + self.ffn(x))

        if self.use_cls_token:
            # Return CLS token representation
            return x[:, 0, :]
        else:
            # Mean pooling over valid positions
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return x


class CardAwareRFTGNet(nn.Module):
    """
    RFTG network that uses learned card embeddings and attention.

    This architecture:
    1. Embeds cards using learned representations
    2. Uses attention to aggregate cards in different zones (hand, tableau)
    3. Combines with non-card features for final prediction

    The network can operate in two modes:
    - 'sparse': Input is (card_ids, zone_ids, other_features)
    - 'dense': Input is original dense format (for backward compatibility)
    """

    CONFIGS = {
        'small': {'hidden_dim': 128, 'num_layers': 2, 'embed_dim': 32, 'num_heads': 4, 'dropout': 0.1},
        'medium': {'hidden_dim': 256, 'num_layers': 4, 'embed_dim': 64, 'num_heads': 4, 'dropout': 0.1},
        'large': {'hidden_dim': 512, 'num_layers': 6, 'embed_dim': 128, 'num_heads': 8, 'dropout': 0.15},
    }

    # Card zones in RFTG
    ZONES = ['hand', 'tableau', 'deck', 'discard']

    def __init__(
        self,
        num_cards: int = 200,
        num_other_features: int = 100,
        num_value_outputs: int = 2,
        num_policy_outputs: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the card-aware network.

        Args:
            num_cards: Number of unique cards in the game
            num_other_features: Number of non-card input features
            num_value_outputs: Value head output dimension
            num_policy_outputs: Policy head output dimension
            hidden_dim: Hidden dimension for residual blocks
            num_layers: Number of residual blocks
            embed_dim: Card embedding dimension
            num_heads: Attention heads for card set encoding
            dropout: Dropout probability
        """
        super().__init__()

        self.num_cards = num_cards
        self.num_other_features = num_other_features
        self.num_value_outputs = num_value_outputs
        self.num_policy_outputs = num_policy_outputs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Card embeddings
        self.card_embedding = CardEmbedding(num_cards, embed_dim)

        # Zone embeddings (to distinguish hand vs tableau vs etc)
        self.zone_embedding = nn.Embedding(len(self.ZONES), embed_dim)

        # Card set encoders for each zone
        self.card_encoders = nn.ModuleDict({
            zone: CardSetEncoder(embed_dim, num_heads, dropout)
            for zone in self.ZONES
        })

        # Combine card representations with other features
        combined_dim = len(self.ZONES) * embed_dim + num_other_features
        self.input_proj = nn.Linear(combined_dim, hidden_dim)

        # Residual backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, 4, dropout)
            for _ in range(num_layers)
        ])

        # Value and policy heads
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_value_outputs),
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_policy_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_config(
        cls,
        config_name: str,
        num_cards: int = 200,
        num_other_features: int = 100,
        num_value_outputs: int = 2,
        num_policy_outputs: int = 7,
    ) -> 'CardAwareRFTGNet':
        """Create from predefined configuration."""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}")

        config = cls.CONFIGS[config_name]
        return cls(
            num_cards=num_cards,
            num_other_features=num_other_features,
            num_value_outputs=num_value_outputs,
            num_policy_outputs=num_policy_outputs,
            **config,
        )

    def forward_sparse(
        self,
        card_ids: torch.Tensor,
        zone_ids: torch.Tensor,
        other_features: torch.Tensor,
        card_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse card representation.

        Args:
            card_ids: Card IDs, shape (batch, max_cards)
            zone_ids: Zone for each card, shape (batch, max_cards)
            other_features: Non-card features, shape (batch, num_other_features)
            card_mask: Valid card mask, shape (batch, max_cards)

        Returns:
            Tuple of (value, policy) probabilities
        """
        batch_size = card_ids.size(0)

        # Get card embeddings
        card_embeds = self.card_embedding(card_ids)  # (batch, max_cards, embed_dim)

        # Add zone information
        zone_embeds = self.zone_embedding(zone_ids)  # (batch, max_cards, embed_dim)
        card_embeds = card_embeds + zone_embeds

        # Encode each zone separately
        zone_representations = []
        for zone_idx, zone in enumerate(self.ZONES):
            # Mask for this zone
            zone_mask = (zone_ids == zone_idx)
            if card_mask is not None:
                zone_mask = zone_mask & card_mask

            # Get cards in this zone
            zone_card_embeds = card_embeds * zone_mask.unsqueeze(-1).float()

            # Encode zone
            zone_repr = self.card_encoders[zone](zone_card_embeds, zone_mask)
            zone_representations.append(zone_repr)

        # Concatenate zone representations with other features
        zone_concat = torch.cat(zone_representations, dim=-1)
        combined = torch.cat([zone_concat, other_features], dim=-1)

        # Project and process through backbone
        x = self.input_proj(combined)
        for block in self.blocks:
            x = block(x)

        # Output heads
        value = F.softmax(self.value_head(x), dim=-1)
        policy = F.softmax(self.policy_head(x), dim=-1)

        return value, policy

    def forward(
        self,
        card_ids: torch.Tensor,
        zone_ids: torch.Tensor,
        other_features: torch.Tensor,
        card_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (alias for forward_sparse).
        """
        return self.forward_sparse(card_ids, zone_ids, other_features, card_mask)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            'num_cards': self.num_cards,
            'num_other_features': self.num_other_features,
            'num_value_outputs': self.num_value_outputs,
            'num_policy_outputs': self.num_policy_outputs,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
        }


def get_architecture(
    arch_name: str,
    num_inputs: int,
    num_outputs: int,
    hidden_dim: Optional[int] = None,
    num_policy_outputs: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to get a network architecture by name.

    Args:
        arch_name: Architecture name. Options:
            - 'baseline': Original 2-layer MLP (RFTGNet)
            - 'residual-small': ResidualMLP small config
            - 'residual-medium': ResidualMLP medium config
            - 'residual-large': ResidualMLP large config
            - 'unified-small': UnifiedRFTGNet small config (value+policy)
            - 'unified-medium': UnifiedRFTGNet medium config (value+policy)
            - 'unified-large': UnifiedRFTGNet large config (value+policy)
        num_inputs: Number of input features
        num_outputs: Number of output classes (value outputs for unified)
        hidden_dim: Hidden dimension (only used for 'baseline')
        num_policy_outputs: Number of policy outputs (only for unified, default 7)

    Returns:
        Configured neural network module
    """
    # Import here to avoid circular dependency
    from rftg_net import RFTGNet

    if arch_name == 'baseline':
        if hidden_dim is None:
            hidden_dim = 50  # Default from original network
        return RFTGNet(num_inputs, hidden_dim, num_outputs)

    elif arch_name.startswith('residual-'):
        config_name = arch_name.replace('residual-', '')
        return ResidualMLP.from_config(config_name, num_inputs, num_outputs)

    elif arch_name.startswith('unified-'):
        config_name = arch_name.replace('unified-', '')
        if num_policy_outputs is None:
            num_policy_outputs = 7  # Default: 7 possible actions in RFTG
        return UnifiedRFTGNet.from_config(
            config_name,
            num_inputs,
            num_value_outputs=num_outputs,
            num_policy_outputs=num_policy_outputs,
        )

    else:
        raise ValueError(f"Unknown architecture: {arch_name}. "
                        f"Available: baseline, residual-small/medium/large, "
                        f"unified-small/medium/large")


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Architecture: {model.__class__.__name__}")

    if hasattr(model, 'num_inputs'):
        print(f"Input dim: {model.num_inputs}")
    if hasattr(model, 'num_hidden') or hasattr(model, 'hidden_dim'):
        hidden = getattr(model, 'hidden_dim', getattr(model, 'num_hidden', None))
        print(f"Hidden dim: {hidden}")
    if hasattr(model, 'num_layers'):
        print(f"Num layers: {model.num_layers}")
    if hasattr(model, 'num_outputs'):
        print(f"Output dim: {model.num_outputs}")

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the architectures
    import sys

    num_inputs = 704  # RFTG eval network input size
    num_outputs = 2   # Win probability for 2 players

    print("Testing modern network architectures for RFTG")
    print(f"Input size: {num_inputs}, Output size: {num_outputs}")

    # Test single-head architectures
    architectures = ['baseline', 'residual-small', 'residual-medium', 'residual-large']

    for arch_name in architectures:
        model = get_architecture(arch_name, num_inputs, num_outputs)
        print_model_summary(model, arch_name)

        # Test forward pass
        x = torch.randn(32, num_inputs)  # Batch of 32
        with torch.no_grad():
            output = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sum (should be ~1.0): {output[0].sum().item():.6f}")
        print()

    # Test unified architectures
    print("\n" + "=" * 60)
    print("Testing unified (value+policy) architectures")
    print("=" * 60)

    unified_architectures = ['unified-small', 'unified-medium', 'unified-large']
    num_policy_outputs = 7

    for arch_name in unified_architectures:
        model = get_architecture(arch_name, num_inputs, num_outputs, num_policy_outputs=num_policy_outputs)
        print_model_summary(model, arch_name)

        # Test forward pass
        x = torch.randn(32, num_inputs)  # Batch of 32
        with torch.no_grad():
            value, policy = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value sum (should be ~1.0): {value[0].sum().item():.6f}")
        print(f"  Policy sum (should be ~1.0): {policy[0].sum().item():.6f}")
        print()

    # Test card embedding modules
    print("\n" + "=" * 60)
    print("Testing card embedding modules")
    print("=" * 60)

    # Test CardEmbedding
    num_cards = 200
    embed_dim = 64
    card_embed = CardEmbedding(num_cards, embed_dim)
    print(f"\nCardEmbedding: {num_cards} cards, {embed_dim} dim")
    print(f"  Parameters: {sum(p.numel() for p in card_embed.parameters()):,}")

    card_ids = torch.randint(2, num_cards + 2, (32, 10))  # batch=32, 10 cards each
    embeds = card_embed(card_ids)
    print(f"  Input shape: {card_ids.shape}")
    print(f"  Output shape: {embeds.shape}")

    # Test CardSetEncoder
    encoder = CardSetEncoder(embed_dim, num_heads=4)
    print(f"\nCardSetEncoder: {embed_dim} dim, 4 heads")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    mask = torch.ones(32, 10, dtype=torch.bool)
    mask[:, 7:] = False  # Mask out last 3 cards
    encoded = encoder(embeds, mask)
    print(f"  Input shape: {embeds.shape}")
    print(f"  Output shape: {encoded.shape}")

    # Test CardAwareRFTGNet
    print("\n" + "=" * 60)
    print("Testing CardAwareRFTGNet architectures")
    print("=" * 60)

    card_aware_configs = ['small', 'medium', 'large']

    for config_name in card_aware_configs:
        model = CardAwareRFTGNet.from_config(
            config_name,
            num_cards=200,
            num_other_features=100,
        )
        print_model_summary(model, f"card-aware-{config_name}")

        # Create test inputs
        batch_size = 32
        max_cards = 20
        card_ids = torch.randint(2, 202, (batch_size, max_cards))
        zone_ids = torch.randint(0, 4, (batch_size, max_cards))
        other_features = torch.randn(batch_size, 100)
        card_mask = torch.ones(batch_size, max_cards, dtype=torch.bool)
        card_mask[:, 15:] = False  # Only 15 valid cards

        with torch.no_grad():
            value, policy = model(card_ids, zone_ids, other_features, card_mask)

        print(f"  Card IDs shape: {card_ids.shape}")
        print(f"  Zone IDs shape: {zone_ids.shape}")
        print(f"  Other features shape: {other_features.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value sum (should be ~1.0): {value[0].sum().item():.6f}")
        print(f"  Policy sum (should be ~1.0): {policy[0].sum().item():.6f}")
        print()
