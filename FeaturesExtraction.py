import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset
from pymfe.mfe import MFE
import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Literal
from sklearn.preprocessing import StandardScaler
import logging

Pooling = Literal["cls", "mean", "max", "token"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class FeaturesExtraction:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_features_from_layer(self, layer: torch.nn.Module, dataset: Dataset, tokenize_fn: Callable, batch_size: int = 16, max_length: int = 128,
                                    device: str = "auto", pooling: Pooling = "cls", return_numpy: bool = False) -> Tuple[Union[torch.Tensor, "np.ndarray"], Optional[Union[torch.Tensor, "np.ndarray"]]]:
        '''
        Extract features from a layer of a model

        Args: 
            layer: The layer of the model to extract features from
            dataset: The dataset to extract features from
            tokenize_fn: The function to tokenize the dataset
            batch_size: The batch size to use for the dataloader
            max_length: The maximum length of the tokens
            device: The device to use for the model
        '''

        # setting up the device and the model
        device = self._resolve_device(device)
        logger.info(f"Using device: {device}")
        self._prepare_model(device)

        # tokenizing the dataset
        tokenized_data = self._tokenize(dataset, tokenize_fn, max_length)
        logger.info(f"Tokenized dataset columns: {list(tokenized_data.features.keys())}")
        loader = self._make_loader(tokenized_data, batch_size)
        logger.info(f"DataLoader ready with batch_size={batch_size}")

        # captured_features is a list of tensors that will be used to store the token-level features captured by the hook [Batch_Size, Sequence_Length, Hidden_Size]
        captured_features: List[torch.Tensor] = []
        # features_chunks is a list of tensors that will be used to store the pooled features [B, H]
        features_chunks: List[torch.Tensor] = []
        # labels_list is a list of tensors that will be used to store the labels [B]
        labels_list: List[torch.Tensor] = []

        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            captured_features.append(out.detach())
        handle = layer.register_forward_hook(hook_fn)

        # hook the layer for each batch
        try:
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", batch.get("label", None))
                if labels is not None:
                    labels_list.append(labels.detach().cpu())

                # clear the captured_features list
                captured_features.clear()

                # forward pass
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

                # can be [B, T, H] or [B, H] depending on the pooling
                h = captured_features[0]
                logger.debug(f"Hidden shape={tuple(h.shape)} | Pooling={pooling}")
                pooled_output = h  # default assignment to satisfy linters
                if h.dim() == 2:
                    pooled_output = h
                elif h.dim() == 3:
                    if pooling == "token":
                        pooled_output = h
                    else:
                        pooled_output = self._pool(h, attention_mask, pooling)

                features_chunks.append(pooled_output.detach().cpu())

        finally:
            handle.remove()

        # concatenate the features chunks
        if not features_chunks:
            features = torch.empty(0)
        else:
            if pooling == "token" and features_chunks[0].dim() == 3:
                features = self._concat_chunks_with_padding(features_chunks)
            else:
                features = torch.cat(features_chunks, dim=0)
        logger.info(f"Extracted features shape: {tuple(features.shape)}")

        labels_tensor: Optional[torch.Tensor] = None
        if labels_list:
            labels_tensor = torch.cat(labels_list, dim=0)

        if return_numpy:
            features = features.detach().cpu().numpy()
            labels_tensor = labels_tensor.detach().cpu(
            ).numpy() if labels_tensor is not None else None
        return features, labels_tensor

    def _resolve_device(self, device: str) -> str:
        '''
        Resolve the device to use for the model

        Args: 
            device: The device to use for the model
        '''
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _prepare_model(self, device: str) -> None:
        '''
        Prepare the model for the task

        Args: 
            device: The device to use for the model
        '''
        self.model.to(device)
        self.model.eval()   

    def _tokenize(self, dataset: Dataset, tokenize_fn: Callable, max_length: int) -> Dataset:
        def _wrap_tokenize(batch):
            return tokenize_fn(self.tokenizer, batch, max_length)

        # keep label/labels if they exist; drop everything else (raw text, idx, etc.)
        keep = {"label", "labels"}
        remove_cols = [c for c in dataset.column_names if c not in keep]

        tokenized = dataset.map(
            _wrap_tokenize,
            batched=True,
            remove_columns=remove_cols,   # <<< KEY LINE
            desc="Tokenizing",
        )
        # IMPORTANT: don't set_format('torch') here; let the collator create tensors
        return tokenized



    def _make_loader(self, tokenized_data: Dataset, batch_size: int) -> DataLoader:
        collator = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        return DataLoader(tokenized_data, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor, how: Pooling) -> torch.Tensor:
        if how == "token":
            return hidden
        if how == "cls":
            return hidden[:, 0, :]
        if how == "mean":
            m = attention_mask.unsqueeze(-1)
            num = (hidden * m).sum(dim=1)
            den = m.sum(dim=1).clamp_min(1)
            return num / den
    
    @staticmethod
    def _concat_chunks_with_padding(chunks: List[torch.Tensor]) -> torch.Tensor:
        if not chunks:
            return torch.empty(0)
        
        max_T = max(t.shape[1] for t in chunks)
        if all(t.shape[1] == max_T for t in chunks):
            return torch.cat(chunks, dim=0)

        padded = []
        for t in chunks: 
            B, T, H = t.shape
            if T == max_T:
                padded.append(t)
            else:
                pad_T = max_T - T
                t_pad = F.pad(t, (0,0,0,pad_T,0,0), mode="constant", value=0.0)
                padded.append(t_pad)
        return torch.cat(padded, dim=0)
    
    @staticmethod
    def resolve_module_by_name(root: torch.nn.Module, name: str)  -> torch.nn.Module:
        """
        Resolve a module by name from a root module

        Args:
            root: The root module to resolve the module from
            name: The name of the module to resolve
        """
        cur = root
        for part in name.split("."):
            cur = getattr(cur, part)
        return cur
    
    def extract_metafeatures(self, X: torch.Tensor, y: torch.Tensor, groups: List[str], summaries: List[str], random_state: int, dataset_name: str) -> pd.DataFrame:
        '''
        Extract metafeatures from a model

        Args: 
            X: The features to extract metafeatures from
            y: The labels to extract metafeatures from
            groups: The groups to extract metafeatures from
            summaries: The summaries to extract metafeatures from
            random_state: The random state to use for the metafeatures
            dataset_name: Name of the dataset
        '''

        # Convert to numpy if needed
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        # standardization
        X_std = StandardScaler().fit_transform(X)
        y = y.astype(int).ravel()

        # validate summaries
        try:
            valid_sum = set(MFE.valid_summary())
        except Exception:
            valid_sum = {"mean", "sd", "median", "min", "max"}
        summaries = [s for s in (summaries or []) if s in valid_sum]
        if not summaries:
            summaries = list(valid_sum)

        # determine group list
        if groups is None or (isinstance(groups, str) and groups.lower() == "all"):
            try:
                group_list = list(MFE.valid_groups())
            except Exception:
                group_list = []
        else:
            group_list = list(groups) if isinstance(
                groups, (list, tuple, set)) else [groups]
        logger.info(f"Meta-feature groups to extract: {group_list}")

        # extract per-group and concatenate
        dfs: list[pd.DataFrame] = []
        if group_list:
            for g in group_list:
                try:
                    mfe = MFE(groups=[g], summary=summaries, random_state=random_state,
                              score="accuracy", num_cv_folds=3, lm_sample_frac=0.5)
                except TypeError:
                    # some versions accept group as string
                    mfe = MFE(groups=g, summary=summaries, random_state=random_state,
                              score="accuracy", num_cv_folds=3, lm_sample_frac=0.5)
                mfe.fit(X_std, y)
                names, values = mfe.extract()
                df_g = pd.DataFrame({"feature": names, "value": values})
                df_g["group"] = g
                dfs.append(df_g)
                logger.info(f"Extracted {len(names)} meta-features for group '{g}'")
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
                columns=["feature", "value", "group"])
        else:
            # fallback: extract without explicit groups and mark as unknown
            mfe = MFE(summary=summaries, random_state=random_state,
                      score="accuracy", num_cv_folds=3, lm_sample_frac=0.5)
            mfe.fit(X_std, y)
            names, values = mfe.extract()
            df = pd.DataFrame(
                {"feature": names, "value": values, "group": "unknown"})

        df["dataset"] = dataset_name
        return df


if __name__ == "__main__":
    # just for testing purposes
    from transformers import RobertaForSequenceClassification, RobertaTokenizer
    from datasets import load_dataset

    def extract_features_from_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    tokenize_fn: Callable,
    target_layer: torch.nn.Module,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu"
):
        model.to(device)
        model.eval()

        def tokenize_batch(batch):
            return tokenize_fn(tokenizer, batch, max_length)

        tokenized_data = dataset.map(tokenize_batch, batched=True)
        tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        dataloader = DataLoader(tokenized_data, batch_size=batch_size)

        extracted_features = []
        all_labels = []

        def hook_fn(module, input, output):
            extracted_features.append(output.detach().cpu())

        hook_handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                model(input_ids=input_ids, attention_mask=attention_mask)
                all_labels.append(labels)

        hook_handle.remove()

        features = torch.cat(extracted_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return features, labels

    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base")

    fe = FeaturesExtraction(roberta_model, roberta_tokenizer)
    rte_dataset = load_dataset("glue", "rte", split="train")   # tem colunas: sentence1, sentence2, label

    def tokenize_rte_longest(tokenizer, batch, max_length):
        # NÃO use return_tensors="pt" aqui (deixe o collator fazer isso)
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="longest",       # ou "max_length" se quiser T fixo
            truncation=True,
            max_length=max_length,
        )

    # Exemplo 1: hook no denso da cabeça de classificação (saída [B, H])
    layer = roberta_model.classifier.dense
    X_class, y_class = fe.extract_features_from_layer(
        layer=layer,
        dataset=rte_dataset,
        tokenize_fn=tokenize_rte_longest,
        batch_size=8,
        max_length=128,
        device="auto",
        pooling="cls",
    )

    # 2) Saída via função "solta"
    X_free, y_free = extract_features_from_layer(
        model=roberta_model,
        tokenizer=roberta_tokenizer,
        dataset=rte_dataset,
        tokenize_fn=tokenize_rte_longest,
        target_layer=layer,
        batch_size=8,
        max_length=128,
        device="cuda",
    )

    # Comparação
    print("Class  ->", tuple(X_class.shape), None if y_class is None else tuple(y_class.shape))
    print("Free   ->", tuple(X_free.shape), None if y_free is None else tuple(y_free.shape))

    def _as_numpy(t):
        if t is None:
            return None
        return t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else t

    a = _as_numpy(X_class)
    b = _as_numpy(X_free)
    if a.shape == b.shape:
        import numpy as _np
        mae = _np.mean(_np.abs(a - b))
        mx = _np.max(_np.abs(a - b))
        print(f"Features diff: mae={mae:.6f} max={mx:.6f}")
    else:
        print("Features have different shapes, cannot compute diff.")

    ya = _as_numpy(y_class)
    yb = _as_numpy(y_free)
    if ya is not None and yb is not None and ya.shape == yb.shape:
        import numpy as _np
        same_labels = _np.array_equal(ya, yb)
        print(f"Labels equal: {same_labels}")
    else:
        print("Labels not comparable (None or different shapes)")

