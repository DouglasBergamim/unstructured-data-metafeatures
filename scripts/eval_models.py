"""
Evaluate fine-tuned RoBERTa models on GLUE validation sets.

Usage:
    python scripts/eval_models.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

MODELS = {
    "sst2": ("data/pretrained_models/roberta_sst2_finetuned",     ["sentence"],               "sst2"),
    "mrpc": ("data/pretrained_models/roberta_mrpc_finetuned",     ["sentence1", "sentence2"], "mrpc"),
    "qnli": ("data/pretrained_models/roberta_qnli_finetuned",     ["question",  "sentence"],  "qnli"),
    "rte":  ("data/pretrained_models/roberta_rte_finetuned_best", ["sentence1", "sentence2"], "rte"),
}

MAX_LENGTH = 128


def evaluate(task, model_dir, fields, glue_task):
    model_path = REPO_ROOT / model_dir
    if not model_path.exists():
        print(f"[{task.upper()}] model not found at {model_path}, skipping.")
        return

    print(f"\n{'─'*45}")
    print(f"  Evaluating: {task.upper()}")
    print(f"  Model     : {model_path}")

    ds = load_dataset("glue", glue_task)["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize(examples):
        return tokenizer(
            *[examples[f] for f in fields],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    cols_to_remove = [c for c in ds.column_names if c != "label"]
    ds_tok = ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
    ds_tok = ds_tok.rename_column("label", "labels")
    ds_tok.set_format("torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval_tmp", report_to="none"),
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    preds_out = trainer.predict(ds_tok)
    preds = np.argmax(preds_out.predictions, axis=1)
    labels = preds_out.label_ids

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")


if __name__ == "__main__":
    for task, (model_dir, fields, glue_task) in MODELS.items():
        evaluate(task, model_dir, fields, glue_task)
    print(f"\n{'─'*45}")
    print("Done.")
