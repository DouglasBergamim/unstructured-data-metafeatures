"""Utility functions and context managers."""

from contextlib import contextmanager
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Iterator, Union, Dict, Optional
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


@contextmanager
def register_hook(layer: torch.nn.Module, hook_fn: Callable) -> Iterator[torch.utils.hooks.RemovableHandle]:
    """Context manager for safe hook registration and removal.
    
    Args:
        layer: PyTorch module to attach hook to
        hook_fn: Hook function with signature (module, input, output)
        
    Yields:
        RemovableHandle for the registered hook
        
    Example:
        >>> with register_hook(model.layer, my_hook_fn) as handle:
        ...     outputs = model(inputs)
        # Hook automatically removed after context
    """
    handle = layer.register_forward_hook(hook_fn)
    try:
        logger.debug(f"Registered hook on {layer.__class__.__name__}")
        yield handle
    finally:
        handle.remove()
        logger.debug(f"Removed hook from {layer.__class__.__name__}")


def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution with timing.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func_name} after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: int,
    name: str = "tensor"
) -> None:
    """Validate tensor dimensionality.
    
    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        name: Name for error messages
        
    Raises:
        ValueError: If tensor has wrong number of dimensions
    """
    if tensor.dim() != expected_dims:
        raise ValueError(
            f"{name} has {tensor.dim()} dimensions, "
            f"expected {expected_dims}. Shape: {tuple(tensor.shape)}"
        )


def setup_logging(level: int = logging.INFO, format_string: str = None) -> None:
    """Configure logging for the package.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set level for package logger
    pkg_logger = logging.getLogger("features_extraction")
    pkg_logger.setLevel(level)


def save_features(
    features: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
    labels: Optional[Union[torch.Tensor, np.ndarray]],
    output_path: str
) -> None:
    """Save features and labels to file.
    
    Supports multiple formats based on file extension:
    - .npz: NumPy compressed format (for single or multiple arrays)
    - .pt: PyTorch format (for tensors)
    - .parquet: Parquet format (converts to DataFrame)
    
    Args:
        features: Features tensor/array or dict of features by layer
        labels: Optional labels tensor/array
        output_path: Path to save features
        
    Example:
        >>> save_features(features, labels, "features.npz")
        >>> save_features(features_by_layer, labels, "all_layers.npz")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    
    # Convert tensors to numpy if needed for npz/parquet
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    if ext == ".npz":
        # Save as compressed numpy archive
        save_dict = {}
        
        if isinstance(features, dict):
            # Multiple layers
            for layer_name, layer_features in features.items():
                save_dict[f"features_{layer_name}"] = to_numpy(layer_features)
        else:
            # Single layer
            save_dict["features"] = to_numpy(features)
        
        if labels is not None:
            save_dict["labels"] = to_numpy(labels)
        
        np.savez_compressed(str(output_path), **save_dict)
        logger.info(f"Saved features to {output_path} (npz format)")
        
    elif ext == ".pt":
        # Save as PyTorch format
        save_dict = {}
        
        if isinstance(features, dict):
            for layer_name, layer_features in features.items():
                key = f"features_{layer_name}"
                save_dict[key] = layer_features if torch.is_tensor(layer_features) else torch.from_numpy(layer_features)
        else:
            save_dict["features"] = features if torch.is_tensor(features) else torch.from_numpy(features)
        
        if labels is not None:
            save_dict["labels"] = labels if torch.is_tensor(labels) else torch.from_numpy(labels)
        
        torch.save(save_dict, str(output_path))
        logger.info(f"Saved features to {output_path} (PyTorch format)")
        
    elif ext == ".parquet":
        # Convert to DataFrame and save as parquet
        if isinstance(features, dict):
            # Multiple layers: create one row per sample per layer
            dfs = []
            for layer_name, layer_features in features.items():
                feat_np = to_numpy(layer_features)
                # Flatten if needed
                if feat_np.ndim > 2:
                    feat_np = feat_np.reshape(feat_np.shape[0], -1)
                
                df = pd.DataFrame(feat_np, columns=[f"feat_{i}" for i in range(feat_np.shape[1])])
                df["layer"] = layer_name
                dfs.append(df)
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            if labels is not None:
                # Repeat labels for each layer
                labels_np = to_numpy(labels).ravel()
                n_layers = len(features)
                df_combined["label"] = np.tile(labels_np, n_layers)
            
            df_combined.to_parquet(str(output_path), index=False)
        else:
            # Single layer
            feat_np = to_numpy(features)
            if feat_np.ndim > 2:
                feat_np = feat_np.reshape(feat_np.shape[0], -1)
            
            df = pd.DataFrame(feat_np, columns=[f"feat_{i}" for i in range(feat_np.shape[1])])
            
            if labels is not None:
                df["label"] = to_numpy(labels).ravel()
            
            df.to_parquet(str(output_path), index=False)
        
        logger.info(f"Saved features to {output_path} (Parquet format)")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npz, .pt, or .parquet")


def save_metafeatures(meta_df: pd.DataFrame, output_path: str) -> None:
    """Save meta-features DataFrame to file.
    
    Supports .parquet and .csv formats.
    
    Args:
        meta_df: DataFrame containing meta-features
        output_path: Path to save meta-features
        
    Example:
        >>> save_metafeatures(meta_df, "metafeatures.parquet")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    
    if ext == ".parquet":
        meta_df.to_parquet(str(output_path), index=False)
        logger.info(f"Saved meta-features to {output_path} (Parquet format, {len(meta_df)} rows)")
    elif ext == ".csv":
        meta_df.to_csv(str(output_path), index=False)
        logger.info(f"Saved meta-features to {output_path} (CSV format, {len(meta_df)} rows)")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .parquet or .csv")
