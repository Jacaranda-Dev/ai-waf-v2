"""
stages/5_training/03_track_b_99m.py
-------------------------------------
Stage 5.3 — Train the 99M-parameter Track B model using the custom
BPE HttpTokenizer from Stage 4.

This is the missing link identified in the critique: while
04_threshold_calibration.py and 05_canary_eval.py referenced
`track_b_99m`, no training script existed for it.

Architecture decisions
----------------------
  * Custom vocab size: the model's embedding matrix is initialised from
    the Track B tokenizer's vocabulary size (not BERT's 30522) — this
    is critical because domain-specific BPE produces a fundamentally
    different token distribution.
  * WafCollator handles the HttpTokenizer's padding scheme.
  * WafClassifier head is identical to Track A for a fair comparison.
  * CheckpointTracker writes models/track_b/99m/latest for Stage 6.

Run:
    python stages/5_training/03_track_b_99m.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import mlflow
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

from checkpoint_utils import CheckpointTracker
from train_utils import WafClassifier, WafCollator, build_optimizer, evaluate, run_epoch

log = get_logger(__name__)

EXPERIMENT_TYPE = "track_b_99m"


# ---------------------------------------------------------------------------
# Tiny Transformer backbone
# A lightweight encoder initialised from scratch with the custom vocab.
# Parameter count targets ~99M, tunable via config (num_layers, hidden_size,
# num_heads, intermediate_size).
# ---------------------------------------------------------------------------

class TinyTransformerEncoder(nn.Module):
    """
    Compact BERT-style encoder with custom vocab embedding.

    Initialised from scratch against the Track B vocabulary — no
    pre-trained weights are loaded here.  Knowledge distillation in
    03b_distill_track_b.py subsequently transfers teacher signal.

    Args:
        vocab_size:        Size of the Track B BPE vocabulary.
        hidden_size:       Embedding + transformer hidden dimension.
        num_layers:        Number of transformer encoder layers.
        num_heads:         Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        max_position_embeddings: Maximum sequence length.
        dropout:           Dropout probability.
    """

    def __init__(
        self,
        vocab_size:              int,
        hidden_size:             int   = 768,
        num_layers:              int   = 6,
        num_heads:               int   = 12,
        intermediate_size:       int   = 3072,
        max_position_embeddings: int   = 512,
        dropout:                 float = 0.1,
        pad_token_id:            int   = 0,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id

        # Embeddings
        self.word_embeddings  = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embeddings   = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embed = nn.Embedding(2, hidden_size)
        self.embed_norm       = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embed_drop       = nn.Dropout(dropout)

        # Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size = hidden_size

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:                  # (B, L, H)
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        token_types = torch.zeros_like(input_ids)

        x = (
            self.word_embeddings(input_ids)
            + self.pos_embeddings(positions)
            + self.token_type_embed(token_types)
        )
        x = self.embed_drop(self.embed_norm(x))

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        pad_mask = attention_mask == 0   # (B, L) bool

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x   # (B, L, H) — [CLS] pooling done in WafClassifier


class TrackB99MModel(nn.Module):
    """TinyTransformerEncoder + WafClassifier head for Track B."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        encoder_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.encoder    = TinyTransformerEncoder(vocab_size=vocab_size, **encoder_cfg)
        self.classifier = WafClassifier(
            self.encoder.hidden_size,
            num_labels,
            dropout=encoder_cfg.get("dropout", 0.1),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        return self.classifier(hidden)

    def get_hidden_states(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return [CLS] embedding without classification — used by distillation."""
        hidden = self.encoder(input_ids, attention_mask)
        return hidden[:, 0, :]   # (B, H)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WafBpeDataset(Dataset):
    """
    Dataset for Track B BPE tokenizer.

    HttpTokenizer.encode() returns a list[int] directly; no HuggingFace
    BatchEncoding is involved.  WafCollator handles padding downstream.
    """

    def __init__(self, path: Path, tokenizer: HttpTokenizer, seq_len: int) -> None:
        table       = pq.read_table(path, columns=["raw", "label"])
        self.texts  = table["raw"].to_pylist()
        self.labels = table["label"].to_pylist()
        self.tok    = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tok.encode(self.texts[idx])[: self.seq_len]
        return {"input_ids": ids, "label": self.labels[idx]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    require_inputs({
        f"{cfg.tokenizer.track_b.output_dir}/tokenizer.json": "run 03_train_custom_bpe.py",
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(cfg.model.track_b_99m.output_dir) / "checkpoint_meta.json",
        args.force, "Stage 5.3 Track B 99M training"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg   = cfg.model.track_b_99m
    tcfg   = cfg.training
    timer  = StepTimer()

    # ------------------------------------------------------------------
    # Load Track B tokenizer — vocab_size drives the embedding table
    # ------------------------------------------------------------------
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        cfg.tokenizer.seq_len,
    )
    vocab_size = tokenizer.vocab_size
    log.info(f"Track B vocab_size={vocab_size} (custom BPE)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    splits = Path(cfg.paths.data_splits)

    with timer.step("setup_data"):
        train_ds = WafBpeDataset(splits / "train.parquet", tokenizer, cfg.tokenizer.seq_len)
        val_ds   = WafBpeDataset(splits / "val.parquet",   tokenizer, cfg.tokenizer.seq_len)

        collator     = WafCollator(tokenizer, seq_len=cfg.tokenizer.seq_len)
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=tcfg.batch_size * 2, shuffle=False,
                                  collate_fn=collator, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Model — initialised from scratch with custom vocab
    # ------------------------------------------------------------------
    encoder_cfg = {
        "hidden_size":             mcfg.get("hidden_size", 768),
        "num_layers":              mcfg.get("num_layers", 6),
        "num_heads":               mcfg.get("num_heads", 12),
        "intermediate_size":       mcfg.get("intermediate_size", 3072),
        "max_position_embeddings": cfg.tokenizer.seq_len,
        "dropout":                 mcfg.get("dropout", 0.1),
        "pad_token_id":            tokenizer.pad_id,
    }

    model = TrackB99MModel(
        vocab_size=vocab_size,
        num_labels=mcfg.num_labels,
        encoder_cfg=encoder_cfg,
    ).to(device)

    n_params = _count_params(model)
    log.info(f"TrackB99MModel: {n_params / 1e6:.1f}M trainable parameters")

    total_steps = len(train_loader) * tcfg.epochs // tcfg.get("accum_steps", 1)
    warmup      = int(total_steps * tcfg.get("warmup_ratio", 0.10))

    # No backbone_lr_multiplier — all weights are randomly initialised
    optimizer, scheduler = build_optimizer(
        model,
        lr=tcfg.get("track_b_lr", tcfg.lr),
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_steps=warmup,
        total_steps=total_steps,
        backbone_lr_multiplier=1.0,
    )

    scaler  = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    tracker = CheckpointTracker(
        experiment_type=EXPERIMENT_TYPE,
        models_root=Path(cfg.paths.models),
        metric="macro_f1",
        mode="max",
        patience=tcfg.get("patience", 5),
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_experiment(cfg.project.name)
    with mlflow.start_run(run_name=EXPERIMENT_TYPE):
        mlflow.log_params({
            "vocab_size":          vocab_size,
            "num_labels":          mcfg.num_labels,
            "hidden_size":         encoder_cfg["hidden_size"],
            "num_layers":          encoder_cfg["num_layers"],
            "num_heads":           encoder_cfg["num_heads"],
            "trainable_params_M":  round(n_params / 1e6, 2),
            "epochs":              tcfg.epochs,
            "batch_size":          tcfg.batch_size,
            "lr":                  tcfg.get("track_b_lr", tcfg.lr),
            "seq_len":             cfg.tokenizer.seq_len,
            "experiment":          EXPERIMENT_TYPE,
        })

        with timer.step("training"):
            for epoch in range(1, tcfg.epochs + 1):
                train_metrics = run_epoch(
                    model, train_loader, optimizer, scheduler, device,
                    accum_steps=tcfg.get("accum_steps", 1),
                    scaler=scaler,
                )
                val_metrics = evaluate(model, val_loader, device, num_labels=mcfg.num_labels)

                log.info(
                    f"Epoch {epoch}/{tcfg.epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"macro_f1={val_metrics['macro_f1']:.4f}"
                )

                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_metrics.items()} |
                    {f"val_{k}":   v for k, v in val_metrics.items()},
                    step=epoch,
                )

                tracker.step(
                    model, val_metrics,
                    meta={"epoch": epoch, "vocab_size": vocab_size,
                          "n_params_M": round(n_params / 1e6, 2)},
                    step=epoch,
                )

                if tracker.should_stop:
                    log.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        log.info(
            f"Training complete. Best macro_f1={tracker.best_value:.5f} "
            f"→ {tracker.best_ckpt}"
        )
        mlflow.log_artifact(str(tracker.best_ckpt / "checkpoint_meta.json"))
        timer.log_mlflow()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Track B 99M model.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())