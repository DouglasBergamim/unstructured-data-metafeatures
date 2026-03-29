"""
Layer-wise classifier accuracy on GLUE RTE using DeBERTa-v3-base representations.

Extracts hidden states from all 12 encoder layers, applies pooling strategies,
trains three classifiers per (layer, pooling) combination, and plots the results.

Usage:
    python scripts/layer_classifier_plot.py
    python scripts/layer_classifier_plot.py --output results/layer_classifier.png
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "RazyDave/deberta-v3-base-finetuned-rte"
MAX_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Linear SVM": LinearSVC(max_iter=5000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
}

POOLING_STRATEGIES = ["cls"]

# Layer caption: 1-indexed (layer 1 … layer 12)
NUM_LAYERS = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> np.ndarray:
    """Apply pooling strategy to hidden states [B, T, H] → [B, H]."""
    if strategy == "cls":
        out = hidden[:, 0, :]
    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        out = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    elif strategy == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, float("-inf"))
        out = hidden.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")
    return out.cpu().numpy()


def extract_all_layers(
    sentences1: list,
    sentences2: list,
    tokenizer,
    model,
    device: torch.device,
) -> dict:
    """
    Returns dict: {layer_idx (0-based): {pooling: np.ndarray [N, H]}}
    """
    # Initialise storage
    layer_pooled = {i: [] for i in range(NUM_LAYERS)}

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(sentences1), BATCH_SIZE), desc="Extracting"):
            s1 = sentences1[start : start + BATCH_SIZE]
            s2 = sentences2[start : start + BATCH_SIZE]
            enc = tokenizer(
                s1, s2,
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            outputs = model(**enc, output_hidden_states=True)
            # hidden_states: tuple of (num_layers+1) tensors [B, T, H]
            # index 0 = embedding layer, 1..12 = encoder layers
            encoder_hidden = outputs.hidden_states[1:]  # 12 tensors

            for layer_idx, hidden in enumerate(encoder_hidden):
                layer_pooled[layer_idx].append(pool(hidden, enc["attention_mask"], "cls"))

    # Concatenate batches
    for layer_idx in range(NUM_LAYERS):
        layer_pooled[layer_idx] = np.concatenate(layer_pooled[layer_idx], axis=0)

    return layer_pooled


def run_experiments(train_features, val_features, train_labels, val_labels) -> pd.DataFrame:
    records = []
    for layer_idx in range(NUM_LAYERS):
        layer_num = layer_idx + 1  # 1-indexed for display
        X_tr = train_features[layer_idx]
        X_val = val_features[layer_idx]
        for clf_name, clf in CLASSIFIERS.items():
            clf.fit(X_tr, train_labels)
            preds = clf.predict(X_val)
            acc = accuracy_score(val_labels, preds)
            f1 = f1_score(val_labels, preds, average="macro")
            records.append(
                dict(
                    layer_idx=layer_idx,
                    layer_num=layer_num,
                    classifier=clf_name,
                    accuracy=acc,
                    f1_score=f1,
                )
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(df: pd.DataFrame, output_path: str | None = None):
    classifiers = list(CLASSIFIERS.keys())
    layer_nums = list(range(1, NUM_LAYERS + 1))

    colors = {
        "Logistic Regression": "#1f77b4",
        "Linear SVM": "#2ca02c",
        "Random Forest": "#d62728",
    }
    markers = {
        "Logistic Regression": "o",
        "Linear SVM": "s",
        "Random Forest": "^",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Layer-wise Classifier Accuracy on RTE", fontsize=14, fontweight="bold")

    for clf_name in classifiers:
        subset = df[df["classifier"] == clf_name].sort_values("layer_num")
        ax.plot(
            subset["layer_num"],
            subset["accuracy"],
            label=clf_name,
            color=colors[clf_name],
            marker=markers[clf_name],
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(layer_nums)
    ax.set_xticklabels([str(n) for n in layer_nums])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0.4, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Path to save the plot (e.g. results/plot.png)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading dataset...")
    ds = load_dataset("glue", "rte")
    train_s1 = ds["train"]["sentence1"]
    train_s2 = ds["train"]["sentence2"]
    train_labels = np.array(ds["train"]["label"])
    val_s1 = ds["validation"]["sentence1"]
    val_s2 = ds["validation"]["sentence2"]
    val_labels = np.array(ds["validation"]["label"])

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)

    print("Extracting train features...")
    train_features = extract_all_layers(train_s1, train_s2, tokenizer, model, device)
    print("Extracting validation features...")
    val_features = extract_all_layers(val_s1, val_s2, tokenizer, model, device)

    print("Running classifier experiments...")
    df = run_experiments(train_features, val_features, train_labels, val_labels)

    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nBest: {best['classifier']} | layer {best['layer_num']} → acc={best['accuracy']:.4f}")

    print("Generating plot...")
    make_plot(df, output_path=args.output)


if __name__ == "__main__":
    main()
