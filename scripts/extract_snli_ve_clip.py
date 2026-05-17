"""Extrai embeddings CLIP do SNLI-VE em quatro representações.

Representações persistidas em um único .npz:
    - features_image    : embedding CLIP da imagem (L2-normalizado)        [N, 512]
    - features_text     : embedding CLIP da hipótese (L2-normalizado)      [N, 512]
    - features_concat   : concat(image, text)                              [N, 1024]
    - features_pairwise : concat(image, text, |image-text|, image*text)   [N, 2048]
    - labels                                                                [N]

Uso:
    python scripts/extract_snli_ve_clip.py --config scripts/configs/clip_snli_ve.yaml

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
# Dataset preparation
# ---------------------------------------------------------------------------

# Labels aceitos e seu mapeamento para inteiros (ordem canônica)
LABEL_MAP: dict[str, int] = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
INVALID_LABELS = {"-", ""}  # valores inválidos presentes em alguns mirrors


def resolve_label(raw_label) -> int | None:
    """Converte um rótulo bruto (string ou int) para inteiro, ou None se inválido."""
    if isinstance(raw_label, int):
        return raw_label if raw_label >= 0 else None
    s = str(raw_label).strip().lower()
    if s in INVALID_LABELS or s == "-1":
        return None
    return LABEL_MAP.get(s)


def balanced_subsample(dataset, n_per_class: int, seed: int, label_col: str):
    """Seleciona n_per_class amostras por rótulo, descartando rótulos inválidos."""
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
    """Garante que o campo de imagem é um PIL.Image RGB."""
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
# CLIP embedding loop
# ---------------------------------------------------------------------------
def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset,
    device: str,
    batch_size: int,
    label_col: str,
    text_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roda o CLIP e devolve (image_emb, text_emb, labels) já em numpy float32.

    Embeddings são L2-normalizados. Linhas cuja imagem não decoda são descartadas
    (com log) — labels e embeddings ficam alinhados ao final.
    """
    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    labels_chunks: list[int] = []

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    logger.info(f"Extraindo embeddings: N={n}, batch_size={batch_size}, batches={n_batches}")

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
                logger.warning(f"  [batch {batch_idx}] rótulo inválido '{raw_lab}', pulando linha")
                images.pop()
                continue
            texts.append(hyp)
            labels.append(label)

        if not images:
            logger.warning(f"  [batch {batch_idx}] nenhuma imagem válida, pulando batch inteiro")
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

        logger.debug(
            f"  [batch {batch_idx}] pixel_values={tuple(pixel_values.shape)}, "
            f"input_ids={tuple(input_ids.shape)}"
        )

        image_emb = model.get_image_features(pixel_values=pixel_values)
        text_emb = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        image_emb = l2_normalize(image_emb)
        text_emb = l2_normalize(text_emb)

        image_chunks.append(image_emb.detach().cpu().to(torch.float32))
        text_chunks.append(text_emb.detach().cpu().to(torch.float32))
        labels_chunks.extend(labels)

        if batch_idx % 10 == 0 or batch_idx == n_batches:
            total_so_far = sum(c.shape[0] for c in image_chunks)
            logger.info(f"  batch {batch_idx}/{n_batches} — amostras acumuladas: {total_so_far}")

    if skipped_images:
        logger.warning(f"Total de imagens descartadas: {skipped_images}")

    logger.info("Concatenando chunks...")
    image_arr = torch.cat(image_chunks, dim=0).numpy()
    text_arr = torch.cat(text_chunks, dim=0).numpy()
    labels_arr = np.asarray(labels_chunks, dtype=np.int64)

    logger.info(
        f"Embeddings finais — image: {image_arr.shape}, "
        f"text: {text_arr.shape}, labels: {labels_arr.shape}"
    )
    return image_arr, text_arr, labels_arr


def build_representations(
    image_arr: np.ndarray, text_arr: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "image": image_arr,
        "text": text_arr,
        "concat": np.concatenate([image_arr, text_arr], axis=1),
        "pairwise": np.concatenate(
            [image_arr, text_arr, np.abs(image_arr - text_arr), image_arr * text_arr],
            axis=1,
        ),
    }


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
    print("Extração de Embeddings CLIP — SNLI-VE")
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
    logger.info(f"  usando coluna de rótulo='{label_col}', texto='{text_col}'")
    subset = balanced_subsample(raw, n_per_class=n_per_class, seed=seed, label_col=label_col)
    int_labels = [resolve_label(v) for v in subset[label_col]]
    unique, counts = np.unique(np.asarray(int_labels), return_counts=True)
    logger.info(f"  subset: {len(subset)} amostras | {dict(zip(unique.tolist(), counts.tolist()))}")

    # Inspeciona o primeiro exemplo para confirmar schema
    sample = subset[0]
    logger.info(
        f"  exemplo[0] — {label_col}='{sample[label_col]}' (→{resolve_label(sample[label_col])}), "
        f"{text_col}='{str(sample[text_col])[:60]}...', "
        f"image type={type(sample['image']).__name__}"
    )

    logger.info(f"Carregando CLIP: {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info("  CLIPProcessor carregado")
    model = CLIPModel.from_pretrained(model_name)
    logger.info("  CLIPModel carregado")

    device = DeviceManager.resolve("auto")
    DeviceManager.prepare_model(model, device)
    logger.info(f"  modelo em device='{device}', modo eval")

    logger.info("Iniciando extração de embeddings...")
    with torch.no_grad():
        image_arr, text_arr, labels_arr = extract_embeddings(
            model=model,
            processor=processor,
            dataset=subset,
            device=device,
            batch_size=batch_size,
            label_col=label_col,
            text_col=text_col,
        )

    img_norms = np.linalg.norm(image_arr, axis=1)
    txt_norms = np.linalg.norm(text_arr, axis=1)
    logger.info(f"  ||image||₂ — mean={img_norms.mean():.4f}, std={img_norms.std():.6f}")
    logger.info(f"  ||text||₂  — mean={txt_norms.mean():.4f}, std={txt_norms.std():.6f}")

    logger.info("Montando representações compostas...")
    representations = build_representations(image_arr, text_arr)
    for name, arr in representations.items():
        logger.info(f"  {name:<10} shape={arr.shape}, dtype={arr.dtype}")

    logger.info(f"Salvando embeddings em {output_path}...")
    save_features(representations, labels_arr, str(output_path))
    logger.info(f"  salvo com sucesso ({output_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n{'=' * 80}")
    print("Concluído com sucesso!")
    print(f"  {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
