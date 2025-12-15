"""Example usage of the refactored features_extraction package."""

from features_extraction import (
    FeaturesExtraction,
    ExtractionConfig,
    MetaFeatureConfig,
    DeviceManager,
)
from features_extraction.utils import setup_logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
import logging

# Setup logging
setup_logging(level=logging.INFO)

# Example 1: Basic feature extraction
def example_basic_extraction():
    """Extract features from a single layer."""
    print("\n=== Example 1: Basic Feature Extraction ===")
    
    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    
    # Load dataset
    dataset = load_dataset("glue", "rte", split="train[:100]")
    
    # Define tokenization function
    def tokenize_rte(tokenizer, batch, max_length):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
    
    # Create extractor
    extractor = FeaturesExtraction(model, tokenizer)
    
    # Configure extraction
    config = ExtractionConfig(
        batch_size=16,
        max_length=128,
        device="auto",
        pooling="cls",
        return_numpy=True
    )
    
    # Extract from classifier layer
    layer = model.classifier.dense
    features, labels = extractor.extract_features_from_layer(
        layer=layer,
        dataset=dataset,
        tokenize_fn=tokenize_rte,
        config=config
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else None}")


# Example 2: Extract all layers
def example_all_layers():
    """Extract features from all model layers."""
    print("\n=== Example 2: All Layers Extraction ===")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    dataset = load_dataset("glue", "rte", split="train[:50]")
    
    def tokenize_rte(tokenizer, batch, max_length):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
    
    extractor = FeaturesExtraction(model, tokenizer)
    config = ExtractionConfig(batch_size=8, pooling="mean")
    
    features_by_layer, labels = extractor.extract_all_layers(
        dataset=dataset,
        tokenize_fn=tokenize_rte,
        config=config
    )
    
    print(f"Extracted {len(features_by_layer)} layers:")
    for layer_name, features in list(features_by_layer.items())[:3]:
        print(f"  {layer_name}: {tuple(features.shape)}")


# Example 3: Extract meta-features
def example_metafeatures():
    """Extract meta-features from all layers."""
    print("\n=== Example 3: Meta-features Extraction ===")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    dataset = load_dataset("glue", "rte", split="train[:100]")
    
    def tokenize_rte(tokenizer, batch, max_length):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
    
    extractor = FeaturesExtraction(model, tokenizer)
    
    # Configure extraction and meta-features
    extraction_config = ExtractionConfig(
        batch_size=16,
        max_length=128,
        pooling="cls"
    )
    
    meta_config = MetaFeatureConfig(
        groups=["statistical", "info-theory"],
        summaries=["mean", "sd"],
        dataset_name="rte_sample",
        layer_filter=r"layer_\d+"  # Only numbered layers
    )
    
    # Extract meta-features
    meta_df = extractor.extract_all_layers_and_metafeatures(
        dataset=dataset,
        tokenize_fn=tokenize_rte,
        extraction_config=extraction_config,
        meta_config=meta_config
    )
    
    print(f"Meta-features shape: {meta_df.shape}")
    print(f"\nSample meta-features:")
    print(meta_df.head(10))
    print(f"\nLayers: {meta_df['layer'].unique()[:5]}")


# Example 4: Device management
def example_device_management():
    """Demonstrate device management."""
    print("\n=== Example 4: Device Management ===")
    
    # Get device info
    info = DeviceManager.get_device_info()
    print("Device information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Resolve device
    device = DeviceManager.resolve("auto")
    print(f"\nAuto-resolved device: {device}")


# Example 5: Different pooling strategies
def example_pooling_strategies():
    """Compare different pooling strategies."""
    print("\n=== Example 5: Pooling Strategies ===")
    
    from features_extraction.pooling import POOLING_STRATEGIES
    
    print("Available pooling strategies:")
    for name, strategy in POOLING_STRATEGIES.items():
        print(f"  - {name}: {strategy.__class__.__name__}")
    
    # Example usage
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    dataset = load_dataset("glue", "rte", split="train[:20]")
    
    def tokenize_rte(tokenizer, batch, max_length):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
    
    extractor = FeaturesExtraction(model, tokenizer)
    layer = model.classifier.dense
    
    for pooling_name in ["cls", "mean", "max"]:
        config = ExtractionConfig(batch_size=8, pooling=pooling_name)
        features, _ = extractor.extract_features_from_layer(
            layer=layer,
            dataset=dataset,
            tokenize_fn=tokenize_rte,
            config=config
        )
        print(f"  {pooling_name}: {tuple(features.shape)}")


if __name__ == "__main__":
    # Run examples (comment out as needed)
    example_basic_extraction()
    example_all_layers()
    example_metafeatures()
    example_device_management()
    example_pooling_strategies()
