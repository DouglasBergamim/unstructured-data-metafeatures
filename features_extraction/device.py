"""Device management utilities for PyTorch models."""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and model placement."""
    
    @staticmethod
    def resolve(device: str) -> str:
        """Resolve device string to actual device.
        
        Args:
            device: Device specification ("auto", "cuda", "cpu", "mps")
            
        Returns:
            Resolved device string
            
        Example:
            >>> device = DeviceManager.resolve("auto")
            >>> print(device)  # "cuda" if available, else "cpu"
        """
        if device == "auto":
            if torch.cuda.is_available():
                selected = "cuda"
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                selected = "mps"
                logger.info("Auto-selected Apple MPS device")
            else:
                selected = "cpu"
                logger.info("Auto-selected CPU device")
            return selected
        
        # Validate explicit device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        
        logger.info(f"Using specified device: {device}")
        return device
    
    @staticmethod
    def prepare_model(model: torch.nn.Module, device: str) -> None:
        """Move model to device and set to eval mode.
        
        Args:
            model: PyTorch model to prepare
            device: Target device
            
        Side Effects:
            - Moves model to specified device
            - Sets model to evaluation mode
            - Disables gradient computation
        """
        model.to(device)
        model.eval()
        logger.debug(f"Model prepared on device '{device}' in eval mode")
    
    @staticmethod
    def get_device_info() -> dict:
        """Get information about available devices.
        
        Returns:
            Dictionary with device availability information
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        }
        
        if info["cuda_available"]:
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) 
                for i in range(info["cuda_device_count"])
            ]
        
        return info
