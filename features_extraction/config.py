"""Configuration dataclasses for feature extraction."""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Literal

Pooling = Literal["cls", "mean", "max", "token"]


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction from transformer layers.
    
    Attributes:
        batch_size: Number of samples per batch during extraction
        max_length: Maximum sequence length for tokenization
        device: Device to use ("auto", "cuda", "cpu", "mps")
        pooling: Pooling strategy for token representations
        return_numpy: Whether to return numpy arrays instead of tensors
        output_path: Optional path to save features (supports .npz, .pt, .parquet)
    """
    batch_size: int = 16
    max_length: int = 128
    device: str = "auto"
    pooling: Pooling = "mean"
    return_numpy: bool = False
    output_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.pooling not in {"cls", "mean", "max", "token"}:
            raise ValueError(f"Invalid pooling strategy: {self.pooling}")
        if self.output_path is not None:
            valid_exts = {".npz", ".pt", ".parquet"}
            from pathlib import Path
            ext = Path(self.output_path).suffix.lower()
            if ext not in valid_exts:
                raise ValueError(f"output_path must have extension {valid_exts}, got {ext}")


@dataclass
class MetaFeatureConfig:
    """Configuration for meta-feature extraction using PyMFE.
    
    Attributes:
        groups: Meta-feature groups to extract ("all" or list of group names)
        summaries: Summary functions to apply (mean, sd, min, max, etc.)
        random_state: Random seed for reproducibility
        dataset_name: Name identifier for the dataset
        token_reduce: How to reduce token-level features ("mean", "max", "cls")
        layer_filter: Optional regex or list to filter specific layers
        sort_numeric: Whether to sort layers numerically
        output_path: Optional path to save meta-features (supports .parquet, .csv)
    """
    groups: Union[List[str], str] = "all"
    summaries: Optional[List[str]] = None
    random_state: int = 42
    dataset_name: str = "unknown"
    token_reduce: str = "mean"
    layer_filter: Optional[Union[List[str], str]] = None
    sort_numeric: bool = True
    output_path: Optional[str] = None
    
    def __post_init__(self):
        """Set default summaries if not provided."""
        if self.summaries is None:
            self.summaries = ["mean", "sd"]
        if self.token_reduce not in {"mean", "max", "cls"}:
            raise ValueError(f"Invalid token_reduce: {self.token_reduce}")
        if self.output_path is not None:
            valid_exts = {".parquet", ".csv"}
            from pathlib import Path
            ext = Path(self.output_path).suffix.lower()
            if ext not in valid_exts:
                raise ValueError(f"output_path must have extension {valid_exts}, got {ext}")
