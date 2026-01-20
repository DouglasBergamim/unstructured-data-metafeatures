"""Core feature extraction class with modular design."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
import re

from .config import ExtractionConfig, MetaFeatureConfig
from .pooling import get_pooling_strategy, POOLING_STRATEGIES
from .device import DeviceManager
from .tokenizer import DatasetTokenizer
from .metafeatures import MetaFeaturesExtractor
from .utils import register_hook, log_execution, validate_tensor_shape, save_features, save_metafeatures

logger = logging.getLogger(__name__)


class FeaturesExtraction:
    """Extract and analyze features from transformer models.
    
    This class provides a interface for:
    - Extracting features from specific layers using hooks
    - Extracting features from all layers
    - Computing meta-features using PyMFE
    - Supporting multiple pooling strategies
    
    Example:
        >>> from features_extraction import FeaturesExtraction, ExtractionConfig
        >>> 
        >>> extractor = FeaturesExtraction(model, tokenizer)
        >>> config = ExtractionConfig(batch_size=32, pooling="mean")
        >>> 
        >>> features, labels = extractor.extract_features_from_layer(
        ...     layer=model.encoder.layer[11],
        ...     dataset=dataset,
        ...     tokenize_fn=my_tokenize_fn,
        ...     config=config
        ... )
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Initialize feature extractor.
        
        Args:
            model: Pretrained transformer model
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device_manager = DeviceManager()
        self.dataset_tokenizer = DatasetTokenizer(tokenizer)
        self.meta_extractor = MetaFeaturesExtractor()
        
        logger.info(
            f"Initialized FeaturesExtraction with "
            f"model={model.__class__.__name__}, "
            f"tokenizer={tokenizer.__class__.__name__}"
        )
    
    @log_execution
    def extract_features_from_layer(
        self,
        layer: torch.nn.Module,
        dataset: Dataset,
        tokenize_fn: Callable,
        config: Optional[ExtractionConfig] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        pooling: Optional[str] = None,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]]]:
        """Extract features from a specific model layer using forward hooks.
        
        Args:
            layer: Target layer to extract features from
            dataset: HuggingFace Dataset with text/labels
            tokenize_fn: Function(tokenizer, batch, max_length) -> dict
            config: Extraction configuration (default: ExtractionConfig())
            batch_size: Override config batch size
            max_length: Override config max length
            device: Override config device
            pooling: Override config pooling
            return_numpy: Override config return_numpy
            
        Returns:
            Tuple of (features, labels) where:
                - features: shape [N, H] or [N, T, H] depending on pooling
                - labels: shape [N] or None if no labels in dataset
        """
        if config is None:
            config = ExtractionConfig()
        
        if batch_size is not None:
            config.batch_size = batch_size
        if max_length is not None:
            config.max_length = max_length
        if device is not None:
            config.device = device
        if pooling is not None:
            config.pooling = pooling
        if return_numpy is not None:
            config.return_numpy = return_numpy
        
        resolved_device = self.device_manager.resolve(config.device)
        self.device_manager.prepare_model(self.model, resolved_device)
        
        # Tokenize dataset
        tokenized = self.dataset_tokenizer.tokenize_dataset(
            dataset, tokenize_fn, config.max_length
        )
        
        # Extract features with hook
        features, labels = self._extract_with_hook(
            layer, tokenized, config, resolved_device
        )
        
        # Save to file if output_path is specified
        if config.output_path is not None:
            save_features(features, labels, config.output_path)
        
        # Convert to numpy if requested
        return self._convert_output(features, labels, config.return_numpy)
    
    def _extract_with_hook(
        self,
        layer: torch.nn.Module,
        dataset: Dataset,
        config: ExtractionConfig,
        device: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features using forward hook.
        
        Args:
            layer: Target layer
            dataset: Tokenized dataset
            config: Extraction configuration
            device: Device string
            
        Returns:
            Tuple of (features_tensor, labels_tensor)
        """
        captured_features: List[torch.Tensor] = []
        features_chunks: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        
        # Dhook function
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            captured_features.append(out.detach())
        
        # Create dataloader
        loader = self.dataset_tokenizer.create_dataloader(
            dataset, config.batch_size
        )
        
        # Get pooling strategy
        pooling_strategy = get_pooling_strategy(config.pooling)
        
        logger.info(
            f"Extracting features: layer={layer.__class__.__name__}, "
            f"pooling={pooling_strategy.name()}, "
            f"batches={len(loader)}"
        )
        
        # Extract with automatic hook cleanup
        with register_hook(layer, hook_fn):
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    batch_features, batch_labels = self._process_batch(
                        batch, device, captured_features, pooling_strategy
                    )
                    features_chunks.append(batch_features)
                    
                    if batch_labels is not None:
                        labels_list.append(batch_labels)
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.debug(f"Processed {batch_idx + 1}/{len(loader)} batches")
        
        # Concatenate all chunks
        features = self._concatenate_features(features_chunks, config.pooling)
        labels = self._concatenate_labels(labels_list)
        
        logger.info(f"Extraction complete: features.shape={tuple(features.shape)}")
        
        return features, labels
    
    def _process_batch(
        self,
        batch: dict,
        device: str,
        captured_features: List[torch.Tensor],
        pooling_strategy
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process a single batch through the model.
        
        Args:
            batch: Batch dictionary
            device: Device string
            captured_features: List to store captured features
            pooling_strategy: Pooling strategy instance
            
        Returns:
            Tuple of (pooled_features, labels)
        """
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", batch.get("label", None))
        
        # Clear previous captures
        captured_features.clear()
        
        # Forward pass (hook will capture features)
        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get captured hidden states
        hidden = captured_features[0]
        
        # Apply pooling if needed
        if hidden.dim() == 2:
            pooled = hidden
        elif hidden.dim() == 3:
            pooled = pooling_strategy.pool(hidden, attention_mask)
        else:
            raise ValueError(f"Unexpected hidden state shape: {tuple(hidden.shape)}")
        
        # Move to CPU
        pooled_cpu = pooled.detach().cpu()
        labels_cpu = labels.detach().cpu() if labels is not None else None
        
        return pooled_cpu, labels_cpu
    
    @staticmethod
    def _concatenate_features(chunks: List[torch.Tensor], pooling: str) -> torch.Tensor:
        """Concatenate feature chunks with padding if needed.
        
        Args:
            chunks: List of feature tensors
            pooling: Pooling method used
            
        Returns:
            Concatenated tensor
        """
        if not chunks:
            return torch.empty(0)
        
        # Token pooling may need padding
        if pooling == "token" and chunks[0].dim() == 3:
            return FeaturesExtraction._concat_with_padding(chunks)
        
        # Standard concatenation
        return torch.cat(chunks, dim=0)
    
    @staticmethod
    def _concat_with_padding(chunks: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate 3D tensors with padding to max sequence length.
        
        Args:
            chunks: List of [B, T, H] tensors
        Returns:
            Concatenated [N, T_max, H] tensor
        OBS:
        B -> Batch Size, T -> Sequence Lenghth, H -> Hidden Size
        N -> Total sample
        """
        if not chunks:
            return torch.empty(0)
        
        max_T = max(t.shape[1] for t in chunks)
        
        # Check if all have same length
        if all(t.shape[1] == max_T for t in chunks):
            return torch.cat(chunks, dim=0)
        
        # Pad to max length
        padded = []
        for t in chunks:
            B, T, H = t.shape
            if T == max_T:
                padded.append(t)
            else:
                pad_T = max_T - T
                t_pad = F.pad(t, (0, 0, 0, pad_T, 0, 0), mode="constant", value=0.0)
                padded.append(t_pad)
        
        return torch.cat(padded, dim=0)
    
    @staticmethod
    def _concatenate_labels(labels_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Concatenate label tensors.
        
        Args:
            labels_list: List of label tensors
            
        Returns:
            Concatenated tensor or None
        """
        if not labels_list:
            return None
        return torch.cat(labels_list, dim=0)
    
    @staticmethod
    def _convert_output(
        features: torch.Tensor,
        labels: Optional[torch.Tensor],
        to_numpy: bool
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]]]:
        """Convert output tensors to numpy if requested.
        
        Args:
            features: Features tensor
            labels: Labels tensor or None
            to_numpy: Whether to convert to numpy
            
        Returns:
            Tuple of (features, labels) in requested format
        """
        if not to_numpy:
            return features, labels
        
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy() if labels is not None else None
        
        return features_np, labels_np
    
    @log_execution
    def extract_all_layers(
        self,
        dataset: Dataset,
        tokenize_fn: Callable,
        config: Optional[ExtractionConfig] = None,
        **kwargs
    ) -> Tuple[Dict[str, Union[torch.Tensor, np.ndarray]], Optional[Union[torch.Tensor, np.ndarray]]]:
        """Extract features from all layers of the model.
        
        Args:
            dataset: HuggingFace Dataset
            tokenize_fn: Tokenization function
            config: Extraction configuration
            **kwargs: Override config parameters
            
        Returns:
            Tuple of (features_by_layer_dict, labels)
        """
        if config is None:
            config = ExtractionConfig()
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        # Setup
        device = self.device_manager.resolve(config.device)
        self.device_manager.prepare_model(self.model, device)
        
        # Tokenize
        tokenized = self.dataset_tokenizer.tokenize_dataset(
            dataset, tokenize_fn, config.max_length
        )
        loader = self.dataset_tokenizer.create_dataloader(
            tokenized, config.batch_size
        )
        
        # Initialize storage
        features_lists: List[List[torch.Tensor]] = []
        layer_names: List[str] = []
        labels_list: List[torch.Tensor] = []
        first_batch = True
        
        pooling_strategy = get_pooling_strategy(config.pooling)
        
        logger.info(f"Extracting all layers with pooling={pooling_strategy.name()}")
        
        # Extract from all layers
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", batch.get("label", None))
                
                if labels is not None:
                    labels_list.append(labels.detach().cpu())
                
                # Get all hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states
                
                # Initialize on first batch
                if first_batch:
                    L = len(hidden_states)
                    layer_names = ["embeddings"] + [f"layer_{i}" for i in range(L - 1)]
                    features_lists = [[] for _ in range(L)]
                    first_batch = False
                    logger.info(f"Found {L} layers: {layer_names[:3]}...{layer_names[-1]}")
                
                # Pool and store each layer
                for i, h in enumerate(hidden_states):
                    pooled = pooling_strategy.pool(h, attention_mask)
                    features_lists[i].append(pooled.detach().cpu())
        
        # Concatenate by layer
        features_by_layer: Dict[str, torch.Tensor] = {}
        for name, chunks in zip(layer_names, features_lists):
            if not chunks:
                features_by_layer[name] = torch.empty(0)
            else:
                features_by_layer[name] = self._concatenate_features(chunks, config.pooling)
        
        labels_tensor = self._concatenate_labels(labels_list)
        
        # Save to file if output_path is specified
        if config.output_path is not None:
            save_features(features_by_layer, labels_tensor, config.output_path)
        
        # Convert to numpy if requested
        if config.return_numpy:
            features_by_layer = {k: v.numpy() for k, v in features_by_layer.items()}
            labels_out = labels_tensor.numpy() if labels_tensor is not None else None
            return features_by_layer, labels_out
        
        return features_by_layer, labels_tensor
    
    @log_execution
    def extract_all_layers_and_metafeatures(
        self,
        dataset: Dataset,
        tokenize_fn: Callable,
        extraction_config: Optional[ExtractionConfig] = None,
        meta_config: Optional[MetaFeatureConfig] = None,
        return_features: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """Complete pipeline: extract all layers and compute meta-features.
        
        Args:
            dataset: HuggingFace Dataset
            tokenize_fn: Tokenization function
            extraction_config: Feature extraction configuration
            meta_config: Meta-feature configuration
            return_features: Whether to return raw features
            **kwargs: Override config parameters
            
        Returns:
            DataFrame of meta-features, optionally with features dict
        """
        if extraction_config is None:
            extraction_config = ExtractionConfig()
        if meta_config is None:
            meta_config = MetaFeatureConfig()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(extraction_config, key) and value is not None:
                setattr(extraction_config, key, value)
            if hasattr(meta_config, key) and value is not None:
                setattr(meta_config, key, value)
        
        # Extract features from all layers
        features_by_layer, y = self.extract_all_layers(
            dataset=dataset,
            tokenize_fn=tokenize_fn,
            config=extraction_config
        )
        
        # Extract meta-features for each layer
        meta_df = self._extract_metafeatures_for_all_layers(
            features_by_layer, y, meta_config
        )
        
        # Save meta-features if output_path is specified
        if meta_config.output_path is not None:
            save_metafeatures(meta_df, meta_config.output_path)
        
        if return_features:
            return meta_df, features_by_layer
        return meta_df
    
    def _extract_metafeatures_for_all_layers(
        self,
        features_by_layer: Dict[str, Union[np.ndarray, torch.Tensor]],
        y: Union[np.ndarray, torch.Tensor],
        config: MetaFeatureConfig
    ) -> pd.DataFrame:
        """Extract meta-features for each layer.
        
        Args:
            features_by_layer: Dict mapping layer names to feature tensors
            y: Labels
            config: Meta-feature configuration
            
        Returns:
            DataFrame with meta-features and layer column
        """
        # Convert labels to numpy
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        y_np = y_np.astype(int).ravel()
        
        # Get and filter layer names
        layer_names = self._prepare_layer_names(
            list(features_by_layer.keys()),
            config.layer_filter,
            config.sort_numeric
        )
        
        logger.info(f"Extracting meta-features for {len(layer_names)} layers")
        
        # Extract for each layer
        dfs = []
        for layer_name in layer_names:
            X = features_by_layer[layer_name]
            
            # Reduce token-level features if needed
            X_2d = self._prepare_features_for_meta(X, config.token_reduce)
            
            # Extract meta-features
            df_layer = self.meta_extractor.extract(
                X=X_2d,
                y=y_np,
                groups=config.groups,
                summaries=config.summaries,
                dataset_name=config.dataset_name
            )
            df_layer["layer"] = layer_name
            dfs.append(df_layer)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=["feature", "value", "group", "layer", "dataset"]
        )
    
    @staticmethod
    def _prepare_layer_names(
        layer_names: List[str],
        layer_filter: Optional[Union[List[str], str]],
        sort_numeric: bool
    ) -> List[str]:
        """Filter and sort layer names by depth 
        
        Args:
            layer_names: All layer names in the order they appear in forward pass
            layer_filter: Regex pattern or list of names
            sort_numeric: Whether to sort by depth (True keeps forward pass order)
            
        Returns:
            Filtered and optionally sorted layer names
        """
        # Apply filter 
        if layer_filter is not None:
            if isinstance(layer_filter, str):
                pat = re.compile(layer_filter)
                layer_names = [n for n in layer_names if pat.search(n)]
            else:
                allowed = set(layer_filter)
                layer_names = [n for n in layer_names if n in allowed]
        
        # Sort by depth (forward pass order) if requested
        if sort_numeric:
            def _layer_depth_key(name: str) -> Tuple[int, str]:
                if name == "embeddings":
                    return (0, name)
                
                # Match layer_N pattern
                m = re.match(r"layer_(\d+)$", name)
                if m:
                    # layer_0 → depth 1, layer_1 → depth 2, etc.
                    return (int(m.group(1)) + 1, name)
                
                # Other layers (classifiers, etc.) go last
                return (10_000, name)
            
            layer_names = sorted(layer_names, key=_layer_depth_key)
        
        return layer_names
    
    @staticmethod
    def _prepare_features_for_meta(
        X: Union[np.ndarray, torch.Tensor],
        token_reduce: str
    ) -> np.ndarray:
        """Prepare features for meta-feature extraction.
        
        Reduces [N, T, H] to [N, H] if needed.
        
        Args:
            X: Feature tensor or array
            token_reduce: Reduction method ("mean", "max", "cls")
            
        Returns:
            2D numpy array [N, H]
        """
        # Convert to numpy
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        
        # Already 2D
        if X.ndim == 2:
            return X
        
        # Reduce 3D to 2D
        if X.ndim == 3:
            if token_reduce == "mean":
                return X.mean(axis=1)
            elif token_reduce == "max":
                return X.max(axis=1)
            elif token_reduce == "cls":
                return X[:, 0, :]
            else:
                raise ValueError(f"Unknown token_reduce: {token_reduce}")
        
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")
    
    @staticmethod
    def resolve_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
        """Resolve a module by dotted name path.
        
        Args:
            root: Root module
            name: Dotted path (e.g., "encoder.layer.0")
            
        Returns:
            Resolved module
            
        Example:
            >>> layer = FeaturesExtraction.resolve_module_by_name(
            ...     model, "roberta.encoder.layer.11"
            ... )
        """
        current = root
        for part in name.split("."):
            current = getattr(current, part)
        return current
