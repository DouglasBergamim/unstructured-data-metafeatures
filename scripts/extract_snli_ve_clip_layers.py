"""Extrai embeddings CLIP camada-a-camada do SNLI-VE.

Pooling por torre (consistente com a forma como o CLIP gera o embedding final):
    - Vision: token CLS (posição 0) em cada hidden state — [B, 197, 768] → [B, 768]
    - Text:   token EOS (posição argmax do input_ids por amostra) — [B, T, 512] → [B, 512]

Embeddings L2-normalizados por linha. Salvo em um único .npz.

Chaves no .npz de saída:
    features_vision_layer_00 … features_vision_layer_12  [N, 768]
    features_text_layer_00   … features_text_layer_12    [N, 512]
    labels                                                [N]

Camada 00 = embedding de entrada; camadas 01–12 = blocos transformer.

Uso:
    python scripts/extract_snli_ve_clip_layers.py \
        --config scripts/configs/clip_snli_ve_layers.yaml

Saída:
    data/embeddings/{output_name}.npz
"""

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from features_extraction import setup_logging, save_features
from features_extraction.device import DeviceManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset preparation (idêntico ao script de embeddings poolados)
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, int] = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
INVALID_LABELS = {"-", ""}


def resolve_label(raw_label) -> int | None:
    if isinstance(raw_label, int):
        return raw_label if raw_label >= 0 else None
    s = str(raw_label).strip().lower()
    if s in INVALID_LABELS or s == "-1":
        return None
    return LABEL_MAP.get(s)


def balanced_subsample(dataset, n_per_class: int, seed: int, label_col: str):
    rng = random.Random(seed)

    label_to_indices: dict[int, list[int]] = defaultdict(list)
    skipped = 0
    for idx, raw in enumerate(dataset[label_col]):
        label = resolve_label(raw)
        if label is None:
            skipped += 1
            continue
        label_to_indices[label].append(idx)

    if skipped:
        logger.info(f"  descartadas {skipped} linhas com rótulo inválido")

    logger.info(f"  classes encontradas: {sorted(label_to_indices.keys())}")
    for lbl, idxs in sorted(label_to_indices.items()):
        name = next((k for k, v in LABEL_MAP.items() if v == lbl), str(lbl))
        logger.info(f"    classe {lbl} ({name}): {len(idxs)} amostras disponíveis")

    selected: list[int] = []
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        if len(indices) < n_per_class:
            raise ValueError(
                f"Classe {label} tem apenas {len(indices)} amostras, "
                f"insuficiente para n_per_class={n_per_class}"
            )
        chosen = rng.sample(indices, n_per_class)
        selected.extend(chosen)
        logger.info(f"  classe {label}: selecionadas {n_per_class} amostras")

    selected.sort()
    logger.info(f"  subset final: {len(selected)} amostras (índices ordenados)")
    return dataset.select(selected)


def coerce_to_pil(image_field) -> Image.Image:
    if isinstance(image_field, Image.Image):
        img = image_field
    else:
        raise TypeError(
            f"Campo de imagem inesperado (tipo {type(image_field).__name__}). "
            f"Use um mirror SNLI-VE que entregue imagens como PIL."
        )
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Layer-wise CLIP embedding loop
# ---------------------------------------------------------------------------
def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_layer_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset,
    device: str,
    batch_size: int,
    label_col: str,
    text_col: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Extrai embeddings CLS de cada camada oculta de ambas as torres CLIP.

    Retorna (vision_layers, text_layers, labels_arr):
        vision_layers: {"layer_00": [N, 768], ..., "layer_12": [N, 768]}
        text_layers:   {"layer_00": [N, 512], ..., "layer_12": [N, 512]}
        labels_arr:    [N] int64
    """
    vision_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    text_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels_list: list[int] = []

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    logger.info(
        f"Extraindo embeddings por camada: N={n}, batch_size={batch_size}, batches={n_batches}"
    )

    skipped_images = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = start // batch_size + 1
        rows = dataset[start:end]

        logger.debug(f"Batch {batch_idx}/{n_batches}: linhas {start}–{end - 1}")

        images: list[Image.Image] = []
        texts: list[str] = []
        labels: list[int] = []
        for img_field, hyp, raw_lab in zip(rows["image"], rows[text_col], rows[label_col]):
            try:
                images.append(coerce_to_pil(img_field))
            except Exception as exc:
                logger.warning(f"  [batch {batch_idx}] pulando imagem inválida: {exc}")
                skipped_images += 1
                continue
            label = resolve_label(raw_lab)
            if label is None:
                logger.warning(
                    f"  [batch {batch_idx}] rótulo inválido '{raw_lab}', pulando linha"
                )
                images.pop()
                continue
            texts.append(hyp)
            labels.append(label)

        if not images:
            logger.warning(f"  [batch {batch_idx}] nenhuma imagem válida, pulando batch")
            continue

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Vision tower — pooling CLS na posição 0
        # hidden_states: tupla de (n_layers+1) tensores [B, 197, 768]
        vision_out = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        for layer_index, hidden_state in enumerate(vision_out.hidden_states):
            cls_vector = l2_normalize(hidden_state[:, 0, :])
            vision_chunks[f"layer_{layer_index:02d}"].append(
                cls_vector.detach().cpu().to(torch.float32)
            )

        # Text tower — pooling no token EOS por amostra
        # CLIP usa input_ids.argmax(dim=-1) (token EOS é o id mais alto).
        # hidden_states: tupla de (n_layers+1) tensores [B, T, 512]
        text_out = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        batch_size_actual = input_ids.shape[0]
        eos_positions = input_ids.to(torch.int).argmax(dim=-1)
        batch_indices = torch.arange(batch_size_actual, device=input_ids.device)
        for layer_index, hidden_state in enumerate(text_out.hidden_states):
            eos_vector = l2_normalize(hidden_state[batch_indices, eos_positions])
            text_chunks[f"layer_{layer_index:02d}"].append(
                eos_vector.detach().cpu().to(torch.float32)
            )

        labels_list.extend(labels)

        if batch_idx % 10 == 0 or batch_idx == n_batches:
            total_so_far = sum(c.shape[0] for c in vision_chunks["layer_00"])
            logger.info(
                f"  batch {batch_idx}/{n_batches} — amostras acumuladas: {total_so_far}"
            )

    if skipped_images:
        logger.warning(f"Total de imagens descartadas: {skipped_images}")

    logger.info("Concatenando chunks...")
    vision_layers = {k: torch.cat(v, dim=0).numpy() for k, v in vision_chunks.items()}
    text_layers = {k: torch.cat(v, dim=0).numpy() for k, v in text_chunks.items()}
    labels_arr = np.asarray(labels_list, dtype=np.int64)

    n_vision = len(vision_layers)
    n_text = len(text_layers)
    first_v = next(iter(vision_layers.values()))
    first_t = next(iter(text_layers.values()))
    logger.info(
        f"Camadas extraídas — vision: {n_vision} × {first_v.shape}, "
        f"text: {n_text} × {first_t.shape}, labels: {labels_arr.shape}"
    )
    return vision_layers, text_layers, labels_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho para o YAML de configuração",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    cfg = load_config(args.config)

    seed = int(cfg["dataset"]["seed"])
    seed_everything(seed)

    model_name = cfg["model"]["name"]
    output_name = cfg["model"]["output_name"]
    hf_id = cfg["dataset"]["hf_id"]
    split = cfg["dataset"]["split"]
    n_per_class = int(cfg["dataset"]["n_per_class"])
    label_col = cfg["dataset"].get("label_col", "label")
    text_col = cfg["dataset"].get("text_col", "hypothesis")
    batch_size = int(cfg["extraction"]["batch_size"])

    output_path = REPO_ROOT / "data" / "embeddings" / f"{output_name}.npz"

    print(f"\n{'=' * 80}")
    print("Extração de Embeddings CLIP por Camada — SNLI-VE")
    print(f"{'=' * 80}")
    print(f"Config:        {args.config}")
    print(f"Modelo:        {model_name}")
    print(f"Dataset:       {hf_id} (split={split})")
    print(f"Por classe:    {n_per_class}")
    print(f"Seed:          {seed}")
    print(f"Saída:         {output_path}")
    print(f"{'=' * 80}\n")

    logger.info(f"Carregando dataset: {hf_id} (split={split})")
    raw = load_dataset(hf_id, split=split)
    logger.info(f"  {len(raw)} linhas no split '{split}'")
    logger.info(f"  colunas disponíveis: {raw.column_names}")

    logger.info(f"Selecionando subset balanceado ({n_per_class} por classe, seed={seed})...")
    subset = balanced_subsample(raw, n_per_class=n_per_class, seed=seed, label_col=label_col)
    int_labels = [resolve_label(v) for v in subset[label_col]]
    unique, counts = np.unique(np.asarray(int_labels), return_counts=True)
    logger.info(
        f"  subset: {len(subset)} amostras | {dict(zip(unique.tolist(), counts.tolist()))}"
    )

    logger.info(f"Carregando CLIP: {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    device = DeviceManager.resolve("auto")
    DeviceManager.prepare_model(model, device)
    logger.info(f"  modelo em device='{device}', modo eval")

    logger.info("Iniciando extração de embeddings por camada...")
    with torch.no_grad():
        vision_layers, text_layers, labels_arr = extract_layer_embeddings(
            model=model,
            processor=processor,
            dataset=subset,
            device=device,
            batch_size=batch_size,
            label_col=label_col,
            text_col=text_col,
        )

    logger.info("Montando dicionário de representações...")
    representations: dict[str, np.ndarray] = {}
    for k, v in vision_layers.items():
        representations[f"vision_{k}"] = v
    for k, v in text_layers.items():
        representations[f"text_{k}"] = v

    for name, arr in representations.items():
        logger.info(f"  {name:<22} shape={arr.shape}, dtype={arr.dtype}")

    logger.info(f"Salvando em {output_path}...")
    save_features(representations, labels_arr, str(output_path))
    logger.info(f"  salvo com sucesso ({output_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n{'=' * 80}")
    print("Concluído com sucesso!")
    print(f"  {output_path}")
    print(f"  vision: {len(vision_layers)} camadas × {next(iter(vision_layers.values())).shape}")
    print(f"  text:   {len(text_layers)} camadas × {next(iter(text_layers.values())).shape}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
