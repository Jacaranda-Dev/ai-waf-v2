"""
stages/5_training/01_track_a_large.py
--------------------------------------
Stage 5.1 — Fine-tune DeBERTa-v3-base (Track A large) on the WAF dataset.

Refactored from the original to:
  * Remove the runtime-injection pattern that script 02 previously relied on
    — each script now owns its complete training logic independently.
  * Integrate CheckpointTracker for atomic saves + best_model symlink, so
    Stage 6 can always reference models/track_a/large/latest.
  * Use explicit MLflow run name tied to this experiment for unambiguous
    run tracking.

Run:
    python stages/5_training/01_track_a_large.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

from checkpoint_utils import CheckpointTracker
from train_utils import WafClassifier, WafCollator, build_optimizer, evaluate, run_epoch

log = get_logger(__name__)

EXPERIMENT_TYPE = "track_a_large"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WafDataset(Dataset):
    """Thin wrapper around a Parquet split for track-agnostic use."""

    def __init__(self, path: Path, tokenizer: Any, seq_len: int) -> None:
        import pyarrow.parquet as pq
        table       = pq.read_table(path, columns=["raw", "label"])
        self.texts  = table["raw"].to_pylist()
        self.labels = table["label"].to_pylist()
        self.tok    = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tok.encode(
            self.texts[idx],
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=True,
        )
        return {"input_ids": ids, "label": self.labels[idx]}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TrackALargeModel(torch.nn.Module):
    """DeBERTa-v3-base backbone + WafClassifier head."""

    def __init__(self, base_model: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(base_model)
        hidden          = self.backbone.config.hidden_size
        self.classifier = WafClassifier(hidden, num_labels, dropout)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa returns last_hidden_state; [CLS] pooling handled in WafClassifier
        return self.classifier(out.last_hidden_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    require_inputs({
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(cfg.model.track_a_large.output_dir) / "checkpoint_meta.json",
        args.force, "Stage 5.1 Track A large training"
    ):
        return

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg    = cfg.model.track_a_large
    tcfg    = cfg.training
    timer   = StepTimer()

    log.info(f"Training {EXPERIMENT_TYPE} | device={device} | base={mcfg.base_model}")

    # ------------------------------------------------------------------
    # Tokenizer + data
    # ------------------------------------------------------------------
    with timer.step("setup_data"):
        tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model)
        splits    = Path(cfg.paths.data_splits)

        train_ds = WafDataset(splits / "train.parquet", tokenizer, cfg.tokenizer.seq_len)
        val_ds   = WafDataset(splits / "val.parquet",   tokenizer, cfg.tokenizer.seq_len)

        collator   = WafCollator(tokenizer, seq_len=cfg.tokenizer.seq_len)
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=tcfg.batch_size * 2, shuffle=False,
                                  collate_fn=collator, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TrackALargeModel(
        base_model=mcfg.base_model,
        num_labels=mcfg.num_labels,
        dropout=mcfg.get("dropout", 0.1),
    ).to(device)

    total_steps = len(train_loader) * tcfg.epochs // tcfg.get("accum_steps", 1)
    warmup      = int(total_steps * tcfg.get("warmup_ratio", 0.06))

    optimizer, scheduler = build_optimizer(
        model,
        lr=tcfg.lr,
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_steps=warmup,
        total_steps=total_steps,
        backbone_lr_multiplier=tcfg.get("backbone_lr_multiplier", 0.1),
    )

    scaler  = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    tracker = CheckpointTracker(
        experiment_type=EXPERIMENT_TYPE,
        models_root=Path(cfg.paths.models),
        metric="macro_f1",
        mode="max",
        patience=tcfg.get("patience", 3),
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_experiment(cfg.project.name)
    with mlflow.start_run(run_name=EXPERIMENT_TYPE):
        mlflow.log_params({
            "base_model":   mcfg.base_model,
            "num_labels":   mcfg.num_labels,
            "epochs":       tcfg.epochs,
            "batch_size":   tcfg.batch_size,
            "lr":           tcfg.lr,
            "seq_len":      cfg.tokenizer.seq_len,
            "experiment":   EXPERIMENT_TYPE,
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
                    meta={"epoch": epoch, "train_loss": train_metrics["loss"]},
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
    p = argparse.ArgumentParser(description=f"Fine-tune {EXPERIMENT_TYPE}.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())