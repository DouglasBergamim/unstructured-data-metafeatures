# 🧠 Unstructured Data Metafeatures

Extraction and analysis of meta-features from transformer neural networks for unstructured data tasks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

## 📋 Overview

This project provides a comprehensive framework for extracting and analyzing meta-features from transformer-based neural networks. It supports:

- **Feature Extraction** from specific layers or all layers
- **Meta-feature Computation** using PyMFE (statistical, information-theoretic, model-based)
- **Fine-tuning** of pre-trained models on custom datasets
- **Multiple Pooling Strategies** (CLS, mean, max, token-level)
- **Modular Architecture** with type safety and comprehensive logging

## 🎯 Key Features

- ✅ **Modular Design**: Separated concerns (pooling, device management, meta-features)
- ✅ **Type Safety**: Full type hints with Python 3.8+ typing
- ✅ **Design Patterns**: Strategy pattern for pooling, context managers for hooks
- ✅ **Flexible Configuration**: Dataclass-based configuration
- ✅ **GPU Support**: CUDA, MPS (Apple Silicon), and CPU
- ✅ **Fine-tuning Integration**: Built-in support for model fine-tuning with HuggingFace Trainer
- ✅ **Comprehensive Logging**: Detailed execution tracking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/unstructured-data-metafeatures.git
cd unstructured-data-metafeatures

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from features_extraction import (
    FeaturesExtraction,
    ExtractionConfig,
    MetaFeatureConfig,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset

# Load model and tokenizer
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

# Load dataset
dataset = load_dataset("glue", "rte", split="train[:100]")

# Define tokenization function
def tokenize_fn(tokenizer, batch, max_length):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

# Extract features
features, labels = extractor.extract_features_from_layer(
    layer=model.classifier.dense,
    dataset=dataset,
    tokenize_fn=tokenize_fn,
    config=config
)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
```

## 📊 Supported Models

The framework supports any HuggingFace transformer model, including:

### Base Models
- **BERT** (`bert-base-uncased`, `bert-large-uncased`)
- **RoBERTa** (`roberta-base`, `roberta-large`)
- **DistilBERT** (`distilbert-base-uncased`)
- **ALBERT** (`albert-base-v2`, `albert-large-v2`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **ELECTRA** (`google/electra-base-discriminator`)

### Multilingual Models
- **mBERT** (`bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`xlm-roberta-base`)
- **BERTimbau** (Portuguese) (`neuralmind/bert-base-portuguese-cased`)

### Domain-Specific Models
- **SciBERT** (`allenai/scibert_scivocab_uncased`)
- **BioBERT** (`dmis-lab/biobert-v1.1`)
- **LegalBERT** (`nlpaueb/legal-bert-base-uncased`)

## 📂 Project Structure

```
unstructured-data-metafeatures/
├── features_extraction/          # Main package
│   ├── __init__.py              # Package exports
│   ├── core.py                  # Main FeaturesExtraction class
│   ├── config.py                # Configuration dataclasses
│   ├── pooling.py               # Pooling strategies
│   ├── device.py                # Device management
│   ├── tokenizer.py             # Dataset tokenization
│   ├── metafeatures.py          # Meta-feature extraction
│   ├── utils.py                 # Utilities and decorators
│   ├── examples.py              # Usage examples
│   └── README.md                # Package documentation
├── Notebooks/                    # Jupyter notebooks
│   ├── test_features_extraction.ipynb    # Complete testing suite
│   ├── features_by_layers.ipynb          # Layer-wise analysis
│   ├── features_extract.ipynb            # Feature extraction examples
│   └── meta_features.ipynb               # Meta-features analysis
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🔧 Configuration

### Extraction Configuration

```python
from features_extraction import ExtractionConfig

config = ExtractionConfig(
    batch_size=32,          # Batch size for processing
    max_length=128,         # Maximum sequence length
    device="auto",          # "auto", "cuda", "mps", or "cpu"
    pooling="mean",         # "cls", "mean", "max", or "token"
    return_numpy=True       # Return numpy arrays instead of tensors
)
```

### Meta-Feature Configuration

```python
from features_extraction import MetaFeatureConfig

meta_config = MetaFeatureConfig(
    groups=["statistical", "info-theory", "model-based"],
    summaries=["mean", "sd"],
    dataset_name="my_dataset",
    token_reduce="mean"
)
```

## 📈 Meta-Features

The framework computes various meta-features using PyMFE:

### Statistical Meta-Features
- Mean, standard deviation, variance
- Skewness, kurtosis
- Min, max, range
- Quartiles, IQR

### Information-Theoretic Meta-Features
- Entropy
- Mutual information
- Attribute correlation

### Model-Based Meta-Features
- Decision tree depth
- Linear discriminant features
- Naive Bayes features

## 🧪 Fine-Tuning

The project includes complete fine-tuning capabilities:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## 📚 Notebooks

### `test_features_extraction.ipynb`
Complete testing suite including:
- Model loading and configuration
- Fine-tuning on RTE dataset
- Feature extraction from specific layers
- Feature extraction from all layers
- Meta-feature computation
- Visualization and analysis

### `features_by_layers.ipynb`
Layer-wise feature analysis for understanding model internals

### `meta_features.ipynb`
Detailed meta-feature extraction and visualization

## 🎨 Pooling Strategies

The framework supports multiple pooling strategies:

- **CLS**: Use the [CLS] token representation
- **Mean**: Average of all token embeddings (excluding padding)
- **Max**: Maximum of all token embeddings (excluding padding)
- **Token**: Keep all token representations (no pooling)

## 🔍 Supported Datasets

### GLUE Benchmark
- RTE (Recognizing Textual Entailment)
- MRPC (Microsoft Research Paraphrase Corpus)
- QQP (Quora Question Pairs)
- MNLI (Multi-Genre Natural Language Inference)
- QNLI (Question Natural Language Inference)
- SST-2 (Stanford Sentiment Treebank)
- CoLA (Corpus of Linguistic Acceptability)

### Other Popular Datasets
- SQuAD (Question Answering)
- IMDB (Sentiment Analysis)
- AG News (News Classification)
- TREC (Question Classification)

## 💻 Device Support

Automatic device selection with fallback:

```python
from features_extraction import DeviceManager

# Auto-detect best device (cuda > mps > cpu)
device = DeviceManager.resolve("auto")

# Get device information
info = DeviceManager.get_device_info()
print(info)
# {
#     'cuda_available': False,
#     'cuda_device_count': 0,
#     'mps_available': True
# }
```

## 📊 Example Results

After fine-tuning on RTE dataset:

| Metric    | Base Model | Fine-tuned | Improvement |
|-----------|------------|------------|-------------|
| Accuracy  | 0.5289     | 0.8123     | +53.6%      |
| Precision | 0.5289     | 0.8245     | +55.9%      |
| Recall    | 1.0000     | 0.8856     | -11.4%      |
| F1-Score  | 0.6914     | 0.8542     | +23.5%      |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyMFE - Python Meta-Feature Extractor](https://github.com/ealcobaca/pymfe)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [PyTorch](https://pytorch.org/)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ at ITA (Instituto Tecnológico de Aeronáutica)**
