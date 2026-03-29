"""
Fine-tune roberta-base sequentially on GLUE tasks: SST-2, MRPC, QNLI.

Each task fine-tunes from the base model (not from the previous task).
The best checkpoint per task is saved to data/pretrained_models/roberta_{task}_finetuned/.

Usage:
    python scripts/finetune_roberta_glue.py
    python scripts/finetune_roberta_glue.py --epochs 5
    python scripts/finetune_roberta_glue.py --tasks sst2 mrpc
    python scripts/finetune_roberta_glue.py --epochs 10 --lr 3e-5 --batch-size 32
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    dataset_key: str        # GLUE task name for load_dataset
    input_fields: List[str] # sentence columns to tokenize
    num_labels: int
    metric_key: str         # primary metric name for reporting
    train_fraction: float = 1.0  # fraction of train split to keep (stratified)


TASKS: Dict[str, TaskConfig] = {
    "sst2": TaskConfig(
        dataset_key="sst2",
        input_fields=["sentence"],
        num_labels=2,
        metric_key="accuracy",
        train_fraction=0.25,
    ),
    "mrpc": TaskConfig(
        dataset_key="mrpc",
        input_fields=["sentence1", "sentence2"],
        num_labels=2,
        metric_key="f1",
    ),
    "qnli": TaskConfig(
        dataset_key="qnli",
        input_fields=["question", "sentence"],
        num_labels=2,
        metric_key="accuracy",
        train_fraction=0.15,
    ),
}

DEFAULT_TASK_ORDER = ["sst2", "mrpc", "qnli"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(task: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "weighted", zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        return {"accuracy": acc, "f1": f1, "mcc": mcc}
    return compute_metrics


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def stratified_sample(dataset, fraction: float, seed: int):
    """Return a stratified fraction of a HuggingFace dataset by label."""
    from collections import defaultdict
    import random

    rng = random.Random(seed)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset["label"]):
        label_to_indices[label].append(idx)

    selected = []
    for label, indices in sorted(label_to_indices.items()):
        k = max(1, round(len(indices) * fraction))
        selected.extend(rng.sample(indices, k))

    selected.sort()
    return dataset.select(selected)


def prepare_dataset(task_cfg: TaskConfig, tokenizer, max_length: int, seed: int = 42):
    log.info("  Loading GLUE/%s ...", task_cfg.dataset_key)
    raw = load_dataset("glue", task_cfg.dataset_key)
    train_raw = raw["train"]
    val_raw = raw["validation"]

    if task_cfg.train_fraction < 1.0:
        train_raw = stratified_sample(train_raw, task_cfg.train_fraction, seed)
        log.info(
            "  Train: %d samples (%.0f%% stratified sample) | Validation: %d samples",
            len(train_raw), task_cfg.train_fraction * 100, len(val_raw),
        )
    else:
        log.info("  Train: %d samples | Validation: %d samples", len(train_raw), len(val_raw))

    def tokenize(examples):
        fields = [examples[f] for f in task_cfg.input_fields]
        return tokenizer(
            *fields,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    cols_to_remove = [c for c in train_raw.column_names if c != "label"]

    train_tok = train_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)
    val_tok = val_raw.map(tokenize, batched=True, remove_columns=cols_to_remove)

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    return train_tok, val_tok


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def finetune_task(
    task_name: str,
    task_cfg: TaskConfig,
    base_model: str,
    num_epochs: int,
    lr: float,
    train_batch: int,
    eval_batch: int,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    patience: int,
    seed: int,
    output_root: Path,
) -> dict:
    log.info("")
    log.info("=" * 70)
    log.info("TASK: %s  |  epochs=%d  |  lr=%s  |  batch=%d", task_name.upper(), num_epochs, lr, train_batch)
    log.info("=" * 70)

    checkpoints_dir = output_root / f"roberta_{task_name}_checkpoints"
    save_dir = output_root / f"roberta_{task_name}_finetuned"

    log.info("Loading tokenizer and model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=task_cfg.num_labels
    )

    log.info("Preparing datasets ...")
    train_ds, val_ds = prepare_dataset(task_cfg, tokenizer, max_length, seed=seed)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=seed,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(task_name),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    log.info("Starting training ...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    log.info("Training finished in %.1f s", elapsed)
    log.info("  train/loss       = %.4f", train_result.training_loss)

    log.info("Running final evaluation ...")
    eval_results = trainer.evaluate()

    primary = task_cfg.metric_key
    log.info("  eval/loss        = %.4f", eval_results["eval_loss"])
    log.info("  eval/accuracy    = %.4f", eval_results.get("eval_accuracy", float("nan")))
    log.info("  eval/f1          = %.4f", eval_results.get("eval_f1", float("nan")))
    log.info("  eval/mcc         = %.4f", eval_results.get("eval_mcc", float("nan")))

    log.info("Saving best model to %s ...", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    log.info("Model saved.")

    return {
        "task": task_name,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_results["eval_loss"],
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "eval_f1": eval_results.get("eval_f1"),
        "eval_mcc": eval_results.get("eval_mcc"),
        "primary_metric": primary,
        "primary_value": eval_results.get(f"eval_{primary}"),
        "elapsed_s": elapsed,
        "save_dir": str(save_dir),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune roberta-base on GLUE tasks.")
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASK_ORDER,
        choices=list(TASKS.keys()),
        help="Tasks to run in order (default: sst2 mrpc qnli)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Train batch size per device (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size per device (default: 32)")
    parser.add_argument("--max-length", type=int, default=128, help="Max tokenization length (default: 128)")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio (default: 0.06)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--base-model", type=str, default="roberta-base", help="HuggingFace model ID (default: roberta-base)")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(REPO_ROOT / "data" / "pretrained_models"),
        help="Root directory for saved models",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)

    log.info("Fine-tuning %s on tasks: %s", args.base_model, " -> ".join(args.tasks))
    log.info("Epochs: %d | LR: %s | Batch: %d | Patience: %d", args.epochs, args.lr, args.batch_size, args.patience)
    log.info("Output root: %s", output_root)

    all_results = []
    total_t0 = time.time()

    for task_name in args.tasks:
        task_cfg = TASKS[task_name]
        result = finetune_task(
            task_name=task_name,
            task_cfg=task_cfg,
            base_model=args.base_model,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch=args.batch_size,
            eval_batch=args.eval_batch_size,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            patience=args.patience,
            seed=args.seed,
            output_root=output_root,
        )
        all_results.append(result)

    # Final summary
    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY  (total time: %.1f s)", total_elapsed)
    log.info("=" * 70)
    log.info("%-8s  %-12s  %-10s  %-10s  %-10s", "Task", "Primary", "Value", "Acc", "F1")
    log.info("-" * 60)
    for r in all_results:
        log.info(
            "%-8s  %-12s  %-10.4f  %-10s  %-10s",
            r["task"],
            r["primary_metric"],
            r["primary_value"] if r["primary_value"] is not None else float("nan"),
            f"{r['eval_accuracy']:.4f}" if r["eval_accuracy"] is not None else "n/a",
            f"{r['eval_f1']:.4f}" if r["eval_f1"] is not None else "n/a",
        )
    log.info("")
    for r in all_results:
        log.info("  %s -> %s", r["task"], r["save_dir"])


if __name__ == "__main__":
    main()
