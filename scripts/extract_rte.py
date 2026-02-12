"""Extrai features e metafeatures de modelos fine-tunados no dataset GLUE RTE.

Uso:
    python scripts/extract_rte.py --config scripts/configs/roberta_rte.yaml
    python scripts/extract_rte.py --config scripts/configs/deberta_rte.yaml

Saídas:
    data/features/{output_name}_allsplits_features.npz
    data/meta-features/{output_name}_rte.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from features_extraction import (
    FeaturesExtraction,
    ExtractionConfig,
    MetaFeatureConfig,
    setup_logging,
)


def load_config(config_path: str) -> dict:
    """Lê o YAML e resolve caminhos locais relativos à raiz do repo."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Se model.path não parecer um modelo HuggingFace (sem "/"), trata como caminho local
    model_path = cfg["model"]["path"]
    if "/" not in model_path or model_path.startswith("data/"):
        cfg["model"]["path"] = str(REPO_ROOT / model_path)
        cfg["model"]["_is_local"] = True
    else:
        cfg["model"]["_is_local"] = False

    return cfg


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def tokenize_rte(tokenizer, batch, max_length):
    """Tokeniza pares de sentenças para extração (RTE-específico)."""
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        padding="longest",
        truncation=True,
        max_length=max_length,
    )


def extract_features(model, tokenizer, dataset, cfg, output_path: str):
    """Extrai features de todas as camadas e salva automaticamente."""
    config = ExtractionConfig(
        batch_size=cfg["extraction"]["batch_size"],
        max_length=cfg["extraction"]["max_length"],
        device="auto",
        pooling=cfg["extraction"]["pooling"],
        return_numpy=True,
        output_path=output_path,
    )

    extractor = FeaturesExtraction(model, tokenizer)
    features_by_layer, labels = extractor.extract_all_layers(
        dataset=dataset,
        tokenize_fn=tokenize_rte,
        config=config,
    )

    return extractor, features_by_layer, labels


def extract_metafeatures(extractor, features_by_layer, labels, cfg, output_path: str, dataset_name: str):
    """Calcula metafeatures a partir das features já extraídas e salva automaticamente."""
    meta_config = MetaFeatureConfig(
        groups=cfg["metafeatures"]["groups"],
        summaries=cfg["metafeatures"]["summaries"],
        dataset_name=dataset_name,
        token_reduce="mean",
        output_path=output_path,
    )

    meta_df = extractor.extract_all_layers_and_metafeatures(
        dataset=features_by_layer,
        labels=labels,
        meta_config=meta_config,
    )

    return meta_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Caminho para o YAML de configuração")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    cfg = load_config(args.config)

    model_path = cfg["model"]["path"]
    output_name = cfg["model"]["output_name"]
    is_local = cfg["model"]["_is_local"]

    # Validar modelo local
    if is_local and not Path(model_path).is_dir():
        sys.exit(
            f"[ERRO] Modelo local não encontrado: {model_path}\n"
            f"Execute 'python scripts/pretrain_rte.py' primeiro."
        )

    # Caminhos de saída
    features_output = REPO_ROOT / "data" / "features" / f"{output_name}_allsplits_features.npz"
    metafeatures_output = REPO_ROOT / "data" / "meta-features" / f"{output_name}_rte.parquet"

    print(f"\n{'='*80}")
    print(f"Extração de Features e Metafeatures - GLUE RTE")
    print(f"{'='*80}")
    print(f"Config:           {args.config}")
    print(f"Modelo:           {model_path}")
    print(f"Tipo:             {'Local' if is_local else 'HuggingFace remoto'}")
    print(f"Nome (saída):     {output_name}")
    print(f"{'='*80}\n")

    # Carregar modelo e tokenizer
    print("Carregando modelo e tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Carregando dataset GLUE RTE (train)...")
    dataset = load_dataset("glue", "rte", split="train")
    print(f"  {len(dataset)} exemplos carregados")

    # Extrair features
    print("\nExtraindo features de todas as camadas...")
    extractor, features_by_layer, labels = extract_features(
        model, tokenizer, dataset, cfg, str(features_output)
    )

    print("\n--- Features extraídas ---")
    for layer_name in sorted(features_by_layer.keys()):
        print(f"  {layer_name}: {features_by_layer[layer_name].shape}")
    print(f"\n✓ Salvo em: {features_output}")

    # Extrair metafeatures
    print("\nExtraindo metafeatures...")
    meta_df = extract_metafeatures(
        extractor, features_by_layer, labels, cfg,
        str(metafeatures_output),
        dataset_name=f"{output_name}_rte",
    )

    print("\n--- Metafeatures ---")
    print(f"  Linhas:   {len(meta_df)}")
    print(f"  Colunas:  {list(meta_df.columns)}")
    layers = sorted(meta_df['layer'].unique().tolist())
    print(f"  Layers:   {layers[:3]}...{layers[-1]} ({len(layers)} camadas)")
    print(f"\n✓ Salvo em: {metafeatures_output}")

    print(f"\n{'='*80}")
    print("Concluído com sucesso!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
