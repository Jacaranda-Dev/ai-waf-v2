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
from pathlib import Path

import mlflow
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from checkpoint_utils import CheckpointTracker
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from track_b_model import TinyTransformerEncoder, TrackB99MModel  # noqa: F401
from train_utils import WafCollator, build_optimizer, evaluate, flatten_metrics, run_epoch

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import check_output, require_inputs
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

EXPERIMENT_TYPE = "track_b_99m"


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
        ids = self.tok.encode(self.texts[idx]).ids[: self.seq_len]
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
        Path(cfg.model.track_b_99m.output_dir) / "latest" / "checkpoint_meta.json",
        args.force, "Stage 5.3 Track B 99M training"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg   = cfg.model.track_b_99m
    tcfg   = cfg.training.teacher
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
        "hidden_size":             mcfg.d_model,
        "num_layers":              mcfg.n_layers,
        "num_heads":               mcfg.n_heads,
        "intermediate_size":       mcfg.d_ff,
        "max_position_embeddings": cfg.tokenizer.seq_len,
        "dropout":                 mcfg.dropout,
        "pad_token_id":            tokenizer.pad_token_id,
    }

    model = TrackB99MModel(
        vocab_size=vocab_size,
        num_labels=mcfg.num_labels,
        encoder_cfg=encoder_cfg,
    ).to(device)

    n_params = _count_params(model)
    log.info(f"TrackB99MModel: {n_params / 1e6:.1f}M trainable parameters")

    epochs = max(1, tcfg.max_steps // max(1, len(train_loader)))

    # No backbone_lr_multiplier — all weights are randomly initialised
    optimizer, scheduler = build_optimizer(
        model,
        lr=tcfg.peak_lr,
        weight_decay=tcfg.weight_decay,
        warmup_steps=tcfg.warmup_steps,
        total_steps=tcfg.max_steps,
        backbone_lr_multiplier=1.0,
    )

    tracker = CheckpointTracker(
        experiment_type=EXPERIMENT_TYPE,
        models_root=Path(cfg.paths.models),
        metric="macro_f1",
        mode="max",
        patience=tcfg.early_stopping_patience,
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    Path(cfg.mlflow.tracking_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.project.name)
    with mlflow.start_run(run_name=EXPERIMENT_TYPE):
        mlflow.log_params({
            "vocab_size":          vocab_size,
            "num_labels":          mcfg.num_labels,
            "hidden_size":         encoder_cfg["hidden_size"],
            "num_layers":          encoder_cfg["num_layers"],
            "num_heads":           encoder_cfg["num_heads"],
            "trainable_params_M":  round(n_params / 1e6, 2),
            "max_steps":           tcfg.max_steps,
            "batch_size":          tcfg.batch_size,
            "lr":                  tcfg.peak_lr,
            "seq_len":             cfg.tokenizer.seq_len,
            "experiment":          EXPERIMENT_TYPE,
        })

        with timer.step("training"):
            epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="ep", dynamic_ncols=True)
            for epoch in epoch_bar:
                train_metrics = run_epoch(
                    model, train_loader, optimizer, scheduler, device,
                    accum_steps=tcfg.grad_accum_steps,
                )
                val_metrics = evaluate(model, val_loader, device, num_labels=mcfg.num_labels)

                epoch_bar.set_postfix(
                    tr_loss=f"{train_metrics['loss']:.4f}",
                    val_loss=f"{val_metrics['loss']:.4f}",
                    f1=f"{val_metrics['macro_f1']:.4f}",
                )
                log.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"macro_f1={val_metrics['macro_f1']:.4f}"
                )

                mlflow.log_metrics(
                    flatten_metrics(train_metrics, "train_") |
                    flatten_metrics(val_metrics,   "val_"),
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