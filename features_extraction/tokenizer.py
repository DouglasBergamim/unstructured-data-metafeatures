"""Dataset tokenization utilities."""

from datasets import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from typing import Callable, Set
import logging

logger = logging.getLogger(__name__)


class DatasetTokenizer:
    """Handles dataset tokenization with configurable column retention."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize tokenizer wrapper.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        logger.debug(f"Initialized DatasetTokenizer with {tokenizer.__class__.__name__}")
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenize_fn: Callable,
        max_length: int,
        keep_columns: Set[str] = None
    ) -> Dataset:
        """Tokenize dataset keeping only specified columns.
        
        Args:
            dataset: HuggingFace Dataset to tokenize
            tokenize_fn: Function with signature (tokenizer, batch, max_length) -> dict
            max_length: Maximum sequence length
            keep_columns: Columns to keep (default: {"label", "labels"})
            
        Returns:
            Tokenized dataset
            
        Example:
            >>> tokenizer = DatasetTokenizer(roberta_tokenizer)
            >>> tokenized = tokenizer.tokenize_dataset(
            ...     dataset=rte_dataset,
            ...     tokenize_fn=my_tokenize_fn,
            ...     max_length=128
            ... )
        """
        if keep_columns is None:
            keep_columns = {"label", "labels"}
        
        def _wrap_tokenize(batch):
            return tokenize_fn(self.tokenizer, batch, max_length)
        
        # Determine columns to remove
        remove_cols = [c for c in dataset.column_names if c not in keep_columns]
        
        logger.info(
            f"Tokenizing dataset: keeping {keep_columns}, "
            f"removing {len(remove_cols)} columns"
        )
        
        tokenized = dataset.map(
            _wrap_tokenize,
            batched=True,
            remove_columns=remove_cols,
            desc="Tokenizing dataset",
        )
        
        logger.info(
            f"Tokenization complete. Dataset has {len(tokenized)} examples, "
            f"columns: {list(tokenized.features.keys())}"
        )
        
        return tokenized
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False
    ) -> DataLoader:
        """Create DataLoader with automatic padding collation.
        
        Args:
            dataset: Tokenized dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader instance
        """
        collator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        
        logger.debug(
            f"Creating DataLoader: batch_size={batch_size}, "
            f"shuffle={shuffle}, dataset_size={len(dataset)}"
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator
        )
