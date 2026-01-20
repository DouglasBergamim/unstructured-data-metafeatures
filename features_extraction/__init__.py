"""
Features Extraction Package for Transformer Models

This package provides modular tools for extracting and analyzing features
from transformer-based models, with support for meta-feature computation.

Architecture:
    - core: Main FeaturesExtraction class
    - config: Configuration dataclasses (ExtractionConfig, MetaFeatureConfig)
    - pooling: Pooling strategies (CLS, mean, max, token)
    - device: Device management utilities
    - tokenizer: Dataset tokenization
    - metafeatures: PyMFE-based meta-feature extraction
    - utils: Helper functions and decorators

Example:
    >>> from features_extraction import FeaturesExtraction, ExtractionConfig
    >>> 
    >>> extractor = FeaturesExtraction(model, tokenizer)
    >>> config = ExtractionConfig(batch_size=32, pooling="mean")
    >>> features, labels = extractor.extract_features_from_layer(
    ...     layer=model.encoder.layer[11],
    ...     dataset=dataset,
    ...     tokenize_fn=tokenize_fn,
    ...     config=config
    ... )
"""

from .core import FeaturesExtraction
from .config import ExtractionConfig, MetaFeatureConfig
from .pooling import PoolingStrategy, POOLING_STRATEGIES, get_pooling_strategy
from .device import DeviceManager
from .metafeatures import MetaFeaturesExtractor
from .tokenizer import DatasetTokenizer
from .utils import setup_logging, save_features, save_metafeatures

__all__ = [
    "FeaturesExtraction",
    "ExtractionConfig",
    "MetaFeatureConfig",
    "PoolingStrategy",
    "POOLING_STRATEGIES",
    "get_pooling_strategy",
    "DeviceManager",
    "MetaFeaturesExtractor",
    "DatasetTokenizer",
    "setup_logging",
    "save_features",
    "save_metafeatures",
]

__version__ = "1.0.0"
__author__ = "Douglas Bergamim"
