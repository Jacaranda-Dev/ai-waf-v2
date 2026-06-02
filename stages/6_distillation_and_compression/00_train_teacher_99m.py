"""
stages/6_distillation_and_compression/00_train_teacher_99m.py
--------------------------------------------------------------
Stage 6.0 — Train the 99M-parameter Track B teacher model.

This script was previously absent, making the entire distillation pipeline
non-functional. The teacher and student MUST share the same tokenizer
vocabulary so that soft-target distributions are comparable.

Run:
    python stages/6_distillation_and_compression/00_train_teacher_99m.py \
        --config config/pipeline.yaml \
        [--resume models/track_b/99m/latest_checkpoint.pt]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__, log_file="reports/train_teacher_99m.log")

# ── Sanity-check: teacher vocab must match tokenizer ─────────────────────────

def _verify_vocab_consistency(teacher_cfg, tokenizer: HttpTokenizer) -> None:
    """Raise early if the config vocab_size doesn't match the loaded tokenizer.

    A mismatch here would cause silent misalignment between teacher soft-target
    distributions and student predictions during distillation.
    """
    if teacher_cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Vocab mismatch: teacher config has vocab_size={teacher_cfg.vocab_size} "
            f"but HttpTokenizer reports {tokenizer.vocab_size}. "
            "Ensure cfg.model.track_b_99m.vocab_size matches the tokenizer."
        )
    log.info(f"Vocab consistency check passed: vocab_size={tokenizer.vocab_size}")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: float,
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    nan_streak = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(input_ids, attention_mask)
            loss = out["loss"] if "loss" in out else nn.CrossEntropyLoss()(out["logits"], labels)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            nan_streak += 1
            if nan_streak >= 20:
                raise RuntimeError(
                    f"Teacher training diverged: 20 consecutive non-finite losses. "
                    "Check LR, data quality, and model dimensions."
                )
            scaler.update()
            continue

        nan_streak = 0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if math.isfinite(grad_norm.item()):
            scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss_val * labels.size(0)
        preds = out["logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(input_ids, attention_mask)
            loss = out["loss"] if "loss" in out else nn.CrossEntropyLoss()(out["logits"], labels)

        total_loss += loss.item() * labels.size(0)
        preds = out["logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    teacher_cfg = cfg.model.track_b_99m
    tcfg        = cfg.training.teacher          # expects keys: epochs, batch_size, lr, weight_decay, grad_clip

    seed_everything(cfg.project.seed)

    require_inputs({
        f"{cfg.tokenizer.track_b.output_dir}/tokenizer.json": "run 03_train_custom_bpe.py",
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(teacher_cfg.output_dir) / "best_99m.pt",
        args.force, "Stage 6.0 Teacher 99M training"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    timer  = StepTimer()

    # ── Tokenizer ─────────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )
    _verify_vocab_consistency(teacher_cfg, tokenizer)

    # ── Model ──────────────────────────────────────
    output_dir = Path(teacher_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from checkpoint: {args.resume}")
        teacher = WafClassifier.load(args.resume, teacher_cfg, map_location=str(device))
    else:
        log.info("Initialising fresh 99M teacher model...")
        teacher = WafClassifier.from_config(teacher_cfg)

    teacher.to(device)
    n_params = teacher.count_parameters()
    log.info(f"Teacher parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
    if not (90_000_000 <= n_params <= 110_000_000):
        log.warning(
            f"Expected ~99M parameters but got {n_params:,}. "
            "Review cfg.model.track_b_99m dimensions."
        )

    # ── Data ──────────────────────────────────────
    collator = WafCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.tokenizer.seq_len,
    )
    splits_dir = cfg.paths.data_splits

    with timer.step("setup_data"):
        train_loader = DataLoader(
            WafDataset(get_split_path(splits_dir, "train"), tokenizer._tok, cfg.tokenizer.seq_len),
            batch_size=tcfg.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            WafDataset(get_split_path(splits_dir, "val"), tokenizer._tok, cfg.tokenizer.seq_len),
            batch_size=tcfg.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

    # ── Optimiser & scheduler ─────────────────────
    optimizer = AdamW(
        teacher.parameters(),
        lr=tcfg.peak_lr,
        weight_decay=tcfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    n_epochs    = max(1, tcfg.max_steps // len(train_loader))
    total_steps = n_epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=tcfg.peak_lr, total_steps=total_steps, pct_start=0.05)
    scaler      = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── MLflow ────────────────────────────────────
    init_experiment(cfg)
    import mlflow
    run_name = f"teacher_99m_{int(time.time())}"

    with mlflow.start_run(run_name=run_name, tags=cfg.mlflow.tags):
        mlflow.log_params({
            "model":       "track_b_99m",
            "n_params":    n_params,
            "epochs":      n_epochs,
            "batch_size":  tcfg.batch_size,
            "lr":          tcfg.peak_lr,
            "vocab_size":  teacher_cfg.vocab_size,
            "seq_len":     cfg.tokenizer.seq_len,
        })

        best_val_loss = float("inf")
        patience_count = 0
        early_stop_patience = tcfg.early_stopping_patience

        with timer.step("training"):
            for epoch in range(1, n_epochs + 1):
                t0 = time.perf_counter()
                train_m = _train_epoch(teacher, train_loader, optimizer, scheduler,
                                       device, scaler, tcfg.grad_clip_norm)
                val_m   = _eval_epoch(teacher, val_loader, device)
                elapsed = time.perf_counter() - t0

                log.info(
                    f"Epoch {epoch:03d}/{n_epochs} | "
                    f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
                    f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} | "
                    f"{elapsed:.1f}s"
                )
                mlflow.log_metrics({
                    "train_loss": train_m["loss"],
                    "train_acc":  train_m["acc"],
                    "val_loss":   val_m["loss"],
                    "val_acc":    val_m["acc"],
                }, step=epoch)

                # Checkpoint every epoch
                ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
                teacher.save(ckpt_path)

                # Best model
                if val_m["loss"] < best_val_loss:
                    best_val_loss  = val_m["loss"]
                    patience_count = 0
                    best_path = output_dir / "best_99m.pt"
                    teacher.save(best_path)
                    log.info(f"  ✓ New best val_loss={best_val_loss:.4f} → saved to {best_path}")
                    mlflow.log_artifact(str(best_path))
                else:
                    patience_count += 1
                    log.info(f"  No improvement ({patience_count}/{early_stop_patience})")
                    if patience_count >= early_stop_patience:
                        log.info("Early stopping triggered.")
                        break

        # Symlink latest for resume convenience
        latest_path = output_dir / "latest_checkpoint.pt"
        latest_path.unlink(missing_ok=True)
        latest_path.symlink_to(best_path.resolve())
        timer.log_mlflow()

    log.info(f"Teacher training complete. Best model: {best_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the 99M Track B teacher model.")
    p.add_argument("--config",  default="config/pipeline.yaml",
                   help="Path to pipeline YAML config.")
    p.add_argument("--resume",  default=None,
                   help="Optional path to checkpoint to resume from.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())