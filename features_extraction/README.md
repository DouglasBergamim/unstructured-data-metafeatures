# Features Extraction Package

Modular and type-safe package for extracting and analyzing features from transformer models.

## 🎯 Features

- **Modular Architecture**: Separated concerns (pooling, device management, meta-features)
- **Type Safety**: Full type hints with Python 3.8+ typing
- **Design Patterns**: Strategy pattern for pooling, context managers for hooks
- **Comprehensive Logging**: Detailed execution tracking with decorators
- **Flexible Configuration**: Dataclass-based configuration
- **Multiple Pooling Strategies**: CLS, mean, max, token-level

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from features_extraction import (
    FeaturesExtraction,
    ExtractionConfig,
    MetaFeatureConfig,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset

# Load model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Create extractor
extractor = FeaturesExtraction(model, tokenizer)

# Configure extraction
config = ExtractionConfig(
    batch_size=16,
    max_length=128,
    device="auto",
    pooling="mean"
)

# Extract features
dataset = load_dataset("glue", "rte", split="train[:100]")

def tokenize_fn(tokenizer, batch, max_length):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

features, labels = extractor.extract_features_from_layer(
    layer=model.classifier.dense,
    dataset=dataset,
    tokenize_fn=tokenize_fn,
    config=config
)
```

## 📚 Architecture

```
features_extraction/
├── __init__.py          # Package exports
├── core.py              # Main FeaturesExtraction class
├── config.py            # Configuration dataclasses
├── pooling.py           # Pooling strategies (Strategy pattern)
├── device.py            # Device management
├── tokenizer.py         # Dataset tokenization
├── metafeatures.py      # Meta-feature extraction
├── utils.py             # Utilities and decorators
└── examples.py          # Usage examples
```

## 🔧 Key Components

### 1. Pooling Strategies (Strategy Pattern)

```python
from features_extraction.pooling import POOLING_STRATEGIES

# Available strategies
strategies = {
    "cls": CLSPooling(),      # First token
    "mean": MeanPooling(),    # Masked average
    "max": MaxPooling(),      # Masked maximum
    "token": TokenPooling(),  # No pooling
}
```

### 2. Device Management

```python
from features_extraction import DeviceManager

# Auto-detect best device
device = DeviceManager.resolve("auto")  # cuda > mps > cpu

# Get device info
info = DeviceManager.get_device_info()
```

### 3. Type-Safe Configuration

```python
from features_extraction import ExtractionConfig, MetaFeatureConfig

extraction_config = ExtractionConfig(
    batch_size=32,
    max_length=256,
    device="cuda",
    pooling="mean",
    return_numpy=True
)

meta_config = MetaFeatureConfig(
    groups=["statistical", "model-based"],
    summaries=["mean", "sd"],
    dataset_name="my_dataset",
    token_reduce="mean"
)
```

### 4. Logging

```python
from features_extraction.utils import setup_logging
import logging

# Configure package logging
setup_logging(level=logging.INFO)

# Decorated functions automatically log execution
# Example output:
# 2025-12-10 10:30:45 [INFO] features_extraction.core: Starting extract_features_from_layer
# 2025-12-10 10:30:52 [INFO] features_extraction.core: Completed extract_features_from_layer in 7.23s
```

## 📖 Examples

See `examples.py` for comprehensive usage examples:

1. **Basic Feature Extraction** - Extract from single layer
2. **All Layers Extraction** - Extract from all model layers
3. **Meta-features** - Compute meta-features with PyMFE
4. **Device Management** - Device detection and management
5. **Pooling Strategies** - Compare different pooling methods

## 🎨 Design Patterns Used

- **Strategy Pattern**: Pluggable pooling strategies
- **Context Manager**: Safe hook registration/cleanup
- **Decorator Pattern**: Logging and timing decorators
- **Dataclass Pattern**: Configuration objects
- **Factory Pattern**: Pooling strategy registry

## 🔍 Type Safety

Full type hints throughout:

```python
def extract_features_from_layer(
    self,
    layer: torch.nn.Module,
    dataset: Dataset,
    tokenize_fn: Callable,
    config: Optional[ExtractionConfig] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]]]:
    ...
```

## 📊 Logging Levels

```python
import logging

# DEBUG: Detailed execution flow
# INFO: Key operations and progress
# WARNING: Recoverable issues
# ERROR: Critical failures
```

## 🤝 Migration from Old Code

```python
# Old API
fe = FeaturesExtraction(model, tokenizer)
features, labels = fe.extract_features_from_layer(
    layer=layer,
    dataset=dataset,
    tokenize_fn=fn,
    batch_size=16,
    max_length=128,
    device="auto",
    pooling="cls",
    return_numpy=False
)

# New API (same result)
config = ExtractionConfig(
    batch_size=16,
    max_length=128,
    device="auto",
    pooling="cls",
    return_numpy=False
)

features, labels = fe.extract_features_from_layer(
    layer=layer,
    dataset=dataset,
    tokenize_fn=fn,
    config=config
)

# Or with parameter overrides
features, labels = fe.extract_features_from_layer(
    layer=layer,
    dataset=dataset,
    tokenize_fn=fn,
    batch_size=16,  # Auto-creates config
    pooling="cls"
)
```

## 📄 License

MIT
