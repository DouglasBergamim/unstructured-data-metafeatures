"""Pooling strategies for token-level representations."""

from abc import ABC, abstractmethod
import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""
    
    @abstractmethod
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling to hidden states.
        
        Args:
            hidden: Token-level hidden states [B, T, H]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Pooled representations [B, H] or [B, T, H] for token pooling
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the pooling strategy."""
        pass


class CLSPooling(PoolingStrategy):
    """Extract only the [CLS] token representation."""
    
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Take the first token (CLS) representation.
        
        Args:
            hidden: [B, T, H] tensor
            attention_mask: [B, T] tensor (unused)
            
        Returns:
            [B, H] tensor containing CLS representations
        """
        return hidden[:, 0, :]
    
    def name(self) -> str:
        return "cls"


class MeanPooling(PoolingStrategy):
    """Average pooling with attention mask."""
    
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean over sequence dimension.
        
        Args:
            hidden: [B, T, H] tensor
            attention_mask: [B, T] tensor
            
        Returns:
            [B, H] tensor containing mean-pooled representations
        """
        # Expand mask to match hidden dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden)
        
        # Sum over valid tokens
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        
        # Count valid tokens (avoid division by zero)
        sum_mask = mask_expanded.sum(dim=1).clamp_min(1e-9)
        
        return sum_hidden / sum_mask
    
    def name(self) -> str:
        return "mean"


class MaxPooling(PoolingStrategy):
    """Max pooling with attention mask."""
    
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked max over sequence dimension.
        
        Args:
            hidden: [B, T, H] tensor
            attention_mask: [B, T] tensor
            
        Returns:
            [B, H] tensor containing max-pooled representations
        """
        # Expand mask and apply to hidden states
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden)
        
        # Set padded positions to very negative value
        hidden_masked = hidden.clone()
        hidden_masked[mask_expanded == 0] = -1e9
        
        # Max pooling
        return hidden_masked.max(dim=1)[0]
    
    def name(self) -> str:
        return "max"


class TokenPooling(PoolingStrategy):
    """Keep all token-level representations (no pooling)."""

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return hidden states without modification.

        Args:
            hidden: [B, T, H] tensor
            attention_mask: [B, T] tensor (unused)

        Returns:
            [B, T, H] tensor (unchanged)
        """
        return hidden

    def name(self) -> str:
        return "token"


class FlattenPooling(PoolingStrategy):
    """Flatten all token representations into a 1D vector per sample."""

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Flatten token representations into 1D.

        Args:
            hidden: [B, T, H] tensor
            attention_mask: [B, T] tensor (unused - all tokens are kept)

        Returns:
            [B, T*H] tensor (flattened)

        Example:
            Input: [32, 128, 768] (batch=32, tokens=128, hidden=768)
            Output: [32, 98304] (32 samples, each with 128*768=98304 features)
        """
        batch_size = hidden.size(0)
        return hidden.reshape(batch_size, -1)

    def name(self) -> str:
        return "flatten"


# Registry of available pooling strategies
POOLING_STRATEGIES: Dict[str, PoolingStrategy] = {
    "cls": CLSPooling(),
    "mean": MeanPooling(),
    "max": MaxPooling(),
    "token": TokenPooling(),
    "flatten": FlattenPooling(),
}


def get_pooling_strategy(name: str) -> PoolingStrategy:
    """Get a pooling strategy by name.
    
    Args:
        name: Name of the pooling strategy
        
    Returns:
        PoolingStrategy instance
        
    Raises:
        ValueError: If pooling strategy name is invalid
    """
    if name not in POOLING_STRATEGIES:
        available = ", ".join(POOLING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown pooling strategy '{name}'. "
            f"Available strategies: {available}"
        )
    return POOLING_STRATEGIES[name]
