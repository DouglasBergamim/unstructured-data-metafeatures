"""Fine-tune roberta-base no dataset GLUE RTE e salva o modelo resultante.

Uso:
    python scripts/pretrain_rte.py

Saída:
    data/pretrained_models/roberta_rte_finetuned_best/
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)

from utils import compute_metrics

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_MODEL = "roberta-base"
OUTPUT_MODEL_DIR = REPO_ROOT / "data" / "pretrained_models" / "roberta_rte_finetuned_best"
CHECKPOINTS_DIR = REPO_ROOT / "data" / "pretrained_models" / "rte_checkpoints"

NUM_EPOCHS = 30
LR = 2e-5
TRAIN_BATCH = 16
EVAL_BATCH = 32
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
SEED = 42


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------
def tokenize_function(examples, tokenizer):
    """Tokeniza pares de sentenças para fine-tuning (cell 14 do notebook)."""
    s1 = ["" if x is None else str(x) for x in examples["sentence1"]]
    s2 = ["" if x is None else str(x) for x in examples["sentence2"]]
    return tokenizer(s1, s2, truncation="only_first", max_length=MAX_LENGTH, padding="max_length")


def prepare_datasets(train_raw, val_raw, tokenizer):
    """Aplica tokenização e ajusta formato para o Trainer."""
    cols_to_remove = ["sentence1", "sentence2", "idx"]

    train_tok = train_raw.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)
    val_tok = val_raw.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)

    for ds in (train_tok, val_tok):
        for col in cols_to_remove:
            if col in ds.column_names:
                ds = ds.remove_columns([col])

    train_tok = train_tok.remove_columns([c for c in cols_to_remove if c in train_tok.column_names])
    val_tok = val_tok.remove_columns([c for c in cols_to_remove if c in val_tok.column_names])

    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    return train_tok, val_tok


def train(model, tokenizer, train_ds, val_ds):
    """Constrói Trainer e executa treino + avaliação."""
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(CHECKPOINTS_DIR / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return trainer, eval_results


def main():
    print("Carregando datasets...")
    train_raw = load_dataset("glue", "rte", split="train")
    val_raw = load_dataset("glue", "rte", split="validation")
    print(f"  train: {len(train_raw)} exemplos | val: {len(val_raw)} exemplos")

    print(f"Carregando modelo base: {BASE_MODEL}")
    tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)
    model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL)

    print("Preparando datasets tokenizados...")
    train_tok, val_tok = prepare_datasets(train_raw, val_raw, tokenizer)

    print("Iniciando fine-tuning...")
    trainer, eval_results = train(model, tokenizer, train_tok, val_tok)

    print(f"\nSalvando modelo em {OUTPUT_MODEL_DIR} ...")
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))

    print("\n--- Resultados da avaliação ---")
    for k, v in sorted(eval_results.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
