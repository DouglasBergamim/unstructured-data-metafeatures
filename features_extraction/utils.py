"""Utility functions and context managers."""

from contextlib import contextmanager
import torch
from typing import Callable, Iterator
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
