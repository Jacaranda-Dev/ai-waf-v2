"""
ai_waf_v2.data.dataset
------------------
PyTorch Dataset wrappers for the  pipeline.

WafDataset      — reads a split Parquet file and tokenises on the fly.
WafDatasetMmap  — memory-mapped version for large datasets that don't
                  fit in RAM (uses PyArrow IPC format).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer


class WafDataset(Dataset):
    """
    Map-style dataset backed by a Parquet file.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the Parquet file for this split (e.g. data/splits/train.parquet).
    tokenizer : HFTokenizer
        A HuggingFace fast Tokenizer with padding and truncation configured.
    seq_len : int
        Maximum sequence length. Sequences are truncated/padded to this length.
    label_col : str
        Column name for the integer label.
    text_col : str
        Column name for the raw HTTP request text.
    attack_class_col : str
        Column name for the attack class string (used for per-class metrics).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: "HFTokenizer",
        seq_len: int = 256,
        label_col: str = "label",
        text_col: str = "raw",
        attack_class_col: str = "attack_class",
    ) -> None:
        self.tokenizer       = tokenizer
        self.seq_len         = seq_len
        self.label_col       = label_col
        self.text_col        = text_col
        self.attack_class_col = attack_class_col

        table = pq.read_table(parquet_path, columns=[text_col, label_col, attack_class_col])
        self._texts        = table[text_col].to_pylist()
        self._labels       = table[label_col].to_pylist()
        self._attack_class = table[attack_class_col].to_pylist()

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text  = self._texts[idx]
        label = self._labels[idx]

        # HuggingFace fast tokenizer returns a BatchEncoding
        enc = self.tokenizer.encode(text)
        ids = enc.ids[: self.seq_len]
        attn = enc.attention_mask[: self.seq_len]

        # WafCollator  pads
        # Pad to seq_len
        # pad_len = self.seq_len - len(ids)
        # ids  = ids  + [self.tokenizer.token_to_id("[PAD]") or 0] * pad_len
        # attn = attn + [0] * pad_len

        return {
            "input_ids":      torch.tensor(ids,   dtype=torch.long),
            "attention_mask": torch.tensor(attn,  dtype=torch.long),
            "labels":         torch.tensor(label, dtype=torch.long),
            # kept for per-class metric computation during eval
            "attack_class":   self._attack_class[idx],
        }

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the loss function.
        Useful when the dataset is imbalanced.

        Returns a 1-D tensor of shape (num_classes,).
        """
        from collections import Counter
        counts = Counter(self._labels)
        n_total = len(self._labels)
        n_classes = max(counts.keys()) + 1
        weights = torch.zeros(n_classes)
        for cls, cnt in counts.items():
            weights[cls] = n_total / (n_classes * cnt)
        return weights


class WafDatasetMmap(Dataset):
    """
    Memory-mapped dataset backed by a PyArrow IPC (Feather v2) file.
    Useful when the full dataset exceeds available RAM.

    Use ``WafDataset.to_ipc(path)`` to convert a Parquet file first
    (helper shown below).
    """

    def __init__(
        self,
        ipc_path: str | Path,
        tokenizer: "HFTokenizer",
        seq_len: int = 256,
        label_col: str = "label",
        text_col: str = "raw",
    ) -> None:
        import pyarrow.ipc as ipc

        self.tokenizer = tokenizer
        self.seq_len   = seq_len

        source        = ipc.open_file(ipc_path)
        table         = source.read_all()
        self._texts   = table[text_col]   # kept as pyarrow ChunkedArray
        self._labels  = table[label_col]

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text  = self._texts[idx].as_py()
        label = self._labels[idx].as_py()

        enc     = self.tokenizer.encode(text)
        ids     = enc.ids[: self.seq_len]
        attn    = enc.attention_mask[: self.seq_len]
        pad_len = self.seq_len - len(ids)
        pad_id  = self.tokenizer.token_to_id("[PAD]") or 0

        return {
            "input_ids":      torch.tensor(ids  + [pad_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor(attn + [0]      * pad_len, dtype=torch.long),
            "labels":         torch.tensor(label, dtype=torch.long),
            
            "attack_class":   self._attack_class[idx],
        }


# ─────────────────────────────────────────────────────────
# Parquet ↔ IPC conversion
# ─────────────────────────────────────────────────────────

def parquet_to_ipc(parquet_path: str | Path, ipc_path: str | Path) -> None:
    """Convert a Parquet file to PyArrow IPC (Feather v2) for mmap access."""
    import pyarrow.feather as feather

    table = pq.read_table(parquet_path)
    feather.write_feather(table, ipc_path, compression="uncompressed")


def get_split_path(splits_dir: str | Path, split: str) -> Path:
    """Return the canonical Parquet path for a given split name."""
    return Path(splits_dir) / f"{split}.parquet"


def load_split_stats(parquet_path: str | Path) -> dict:
    """
    Return basic statistics for a Parquet split without loading all data.

    Returns
    -------
    dict with keys: n_total, n_benign, n_malicious, attack_class_counts
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["label", "attack_class"])
    counts = df["label"].value_counts().to_dict()
    return {
        "n_total":     len(df),
        "n_benign":    int(counts.get(0, 0)),
        "n_malicious": int(counts.get(1, 0)),
        "attack_class_counts": df["attack_class"].value_counts().to_dict(),
    }