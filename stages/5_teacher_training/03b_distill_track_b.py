"""
stages/5_training/03b_distill_track_b.py
------------------------------------------
Stage 5.3b — Knowledge distillation: frozen DeBERTa-v3 teacher →
Track B 99M student.

Background
----------
Standard cross-entropy training (script 03) treats each class label as
a one-hot signal.  Distillation additionally transfers the *soft targets*
from the teacher's output distribution — a richer supervision signal that
encodes the teacher's uncertainty and inter-class similarity.

Loss function (Hinton et al., 2015)
-------------------------------------
    L = α · CE(student_logits, hard_labels)
      + (1 − α) · T² · KL(
            softmax(student_logits / T) ‖ softmax(teacher_logits / T)
        )

where:
  α  — interpolation weight (0 = pure distillation, 1 = pure CE).
  T  — temperature; higher values soften the teacher distribution,
       revealing more inter-class information.

Teacher
-------
The fine-tuned DeBERTa-v3-base from Stage 5.1 is loaded frozen.
To preserve the teacher's accuracy, its weights are NOT updated during
this stage.  The teacher runs with `torch.no_grad()`.

Cross-tokenizer alignment
--------------------------
Teacher and student use *different* tokenizers (WordPiece vs BPE).
The teacher is called on *its own* encoding of the same raw text;
the student on the BPE encoding.  Logit shapes match because both models
output (batch, num_labels).

Run:
    python stages/5_training/03b_distill_track_b.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import math
import warnings
from pathlib import Path
from typing import Any

import mlflow
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# DeBERTa teacher uses F.scaled_dot_product_attention which emits this under no_grad()
# Non-determinism is irrelevant since the teacher is frozen and produces no gradients.
warnings.filterwarnings("ignore", message="Memory Efficient attention defaults")

from checkpoint_utils import CheckpointTracker, load_checkpoint, resolve_checkpoint
from track_b_model import TrackB99MModel
from train_utils import WafClassifier, build_optimizer, evaluate, flatten_metrics

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

EXPERIMENT_TYPE = "track_b_99m"   # Overwrites the supervised checkpoint


# ---------------------------------------------------------------------------
# Distillation dataset
# Returns two tokenisations per sample: one for the student (BPE),
# one for the teacher (WordPiece) — handled in the collator.
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """
    Dual-tokenised dataset for teacher-student distillation.

    Each item carries:
      student_ids   — BPE-encoded input for the Track B model.
      teacher_ids   — WordPiece-encoded input for the DeBERTa teacher.
      label         — Hard ground-truth integer class.
    """

    def __init__(
        self,
        path:             Path,
        student_tok:      HttpTokenizer,
        teacher_tok:      Any,
        seq_len:          int,
    ) -> None:
        table        = pq.read_table(path, columns=["raw", "label"])
        self.texts   = table["raw"].to_pylist()
        self.labels  = table["label"].to_pylist()
        self.s_tok   = student_tok
        self.t_tok   = teacher_tok
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        student_ids = self.s_tok.encode(text).ids[: self.seq_len]
        teacher_ids = self.t_tok.encode(
            text, truncation=True, max_length=self.seq_len, add_special_tokens=True
        )
        return {
            "student_ids": student_ids,
            "teacher_ids": teacher_ids,
            "label":       self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Dual collator
# ---------------------------------------------------------------------------

def distill_collate_fn(
    batch:       list[dict],
    student_pad: int,
    teacher_pad: int,
    seq_len:     int,
) -> dict[str, torch.Tensor]:
    """Pad student and teacher sequences independently."""

    def _pad(ids_list: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for ids in ids_list)
        padded, masks = [], []
        for ids in ids_list:
            p = max_len - len(ids)
            padded.append(ids + [pad_id] * p)
            masks.append([1] * len(ids) + [0] * p)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks,  dtype=torch.long),
        )

    s_ids, s_mask = _pad([b["student_ids"] for b in batch], student_pad)
    t_ids, t_mask = _pad([b["teacher_ids"] for b in batch], teacher_pad)
    labels        = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "student_input_ids":      s_ids,
        "student_attention_mask": s_mask,
        "teacher_input_ids":      t_ids,
        "teacher_attention_mask": t_mask,
        "labels":                 labels,
    }


# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------

class FrozenDeBERTaTeacher(nn.Module):
    """
    Fine-tuned DeBERTa-v3 loaded from the Track A large checkpoint.

    All parameters are frozen; the model only performs forward passes
    to supply soft targets for distillation.
    """

    def __init__(self, base_model: str, num_labels: int, ckpt_path: Path) -> None:
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(base_model)
        hidden          = self.backbone.config.hidden_size
        self.classifier = WafClassifier(hidden, num_labels)
        load_checkpoint(ckpt_path, self)

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state)


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels:         torch.Tensor,
    temperature:    float = 4.0,
    alpha:          float = 0.5,
) -> torch.Tensor:
    """
    Weighted sum of cross-entropy (hard labels) and KL divergence (soft targets).

    Args:
        student_logits: (B, C) raw logits from the student.
        teacher_logits: (B, C) raw logits from the (frozen) teacher.
        labels:         (B,)  integer ground-truth class indices.
        temperature:    Softening temperature T.  Typical values: 3–6.
        alpha:          Weight on the CE term.  (1-alpha) weights the KL term.

    Returns:
        Scalar loss tensor with gradient attached to student parameters only.
    """
    T = temperature

    # Hard-label cross-entropy
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft-target KL divergence
    # KLDivLoss expects log-probabilities for input, probabilities for target
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs     = F.softmax(teacher_logits / T,     dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Scale KL by T² to preserve gradient magnitude under temperature scaling
    return alpha * ce_loss + (1.0 - alpha) * (T ** 2) * kl_loss


# ---------------------------------------------------------------------------
# Training loop (distillation-aware)
# ---------------------------------------------------------------------------

def run_distill_epoch(
    student:     nn.Module,
    teacher:     nn.Module,
    loader:      DataLoader,
    optimizer:   Any,
    scheduler:   Any,
    device:      torch.device,
    temperature: float,
    alpha:       float,
    accum_steps: int = 1,
    scaler:      Any = None,   # unused; kept for call-site compat
) -> dict[str, float]:

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    student.train()
    teacher.eval()

    total_loss = total_ce = total_kl = 0.0
    nan_streak = 0
    T = temperature

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        s_ids  = batch["student_input_ids"].to(device)
        s_mask = batch["student_attention_mask"].to(device)
        t_ids  = batch["teacher_input_ids"].to(device)
        t_mask = batch["teacher_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_ctx:
            student_logits = student(input_ids=s_ids, attention_mask=s_mask)
            teacher_logits = teacher(input_ids=t_ids, attention_mask=t_mask)

            ce_loss = F.cross_entropy(student_logits, labels)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits.float() / T, dim=-1),
                reduction="batchmean",
            )
            loss = (alpha * ce_loss + (1.0 - alpha) * T ** 2 * kl_loss) / accum_steps

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            nan_streak += 1
            if nan_streak >= 20:
                raise RuntimeError(
                    "Distillation diverged: 20 consecutive non-finite losses. "
                    "Check LR, temperature, and data quality."
                )
            optimizer.zero_grad()
            continue

        nan_streak = 0
        loss.backward()

        if (i + 1) % accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if not math.isfinite(grad_norm.item()):
                optimizer.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss_val * accum_steps
        total_ce   += ce_loss.item()
        total_kl   += kl_loss.item()

    n = max(len(loader), 1)
    return {
        "loss":    round(total_loss / n, 5),
        "ce_loss": round(total_ce   / n, 5),
        "kl_loss": round(total_kl   / n, 5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    require_inputs({
        f"{cfg.model.track_a_large.output_dir}/latest/checkpoint_meta.json": "run 01_track_a_large.py",
        f"{cfg.tokenizer.track_b.output_dir}/tokenizer.json":                "run 03_train_custom_bpe.py",
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    # No check_output here: 03b shares output_dir with 03 and intentionally
    # overwrites the latest symlink. Use --force on 03 to re-run supervised
    # training; re-running 03b is always safe.

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg    = cfg.model.track_b_99m
    tcfg    = cfg.training.distillation
    timer   = StepTimer()

    temperature = tcfg.temperature
    alpha       = tcfg.alpha_hard

    log.info(
        f"Distillation | T={temperature} | α={alpha} | device={device}"
    )

    # ------------------------------------------------------------------
    # Tokenizers
    # ------------------------------------------------------------------
    student_tok = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        cfg.tokenizer.seq_len,
    )
    teacher_tok = AutoTokenizer.from_pretrained(cfg.model.track_a_large.base_model)

    # ------------------------------------------------------------------
    # Data — dual-tokenised
    # ------------------------------------------------------------------
    splits   = Path(cfg.paths.data_splits)
    seq_len  = cfg.tokenizer.seq_len

    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = DistillDataset(
            splits / f"{split}.parquet",
            student_tok, teacher_tok, seq_len,
        )
        return DataLoader(
            ds,
            batch_size=tcfg.batch_size,  # distillation batch_size
            shuffle=shuffle,
            collate_fn=lambda b: distill_collate_fn(
                b,
                student_pad=student_tok.pad_token_id,
                teacher_pad=teacher_tok.pad_token_id,
                seq_len=seq_len,
            ),
            num_workers=4,
            pin_memory=True,
        )

    with timer.step("setup_data"):
        train_loader = _make_loader("train", shuffle=True)
        val_loader   = _make_loader("val",   shuffle=False)

    # ------------------------------------------------------------------
    # Teacher — loaded from Track A large checkpoint, then frozen
    # ------------------------------------------------------------------
    with timer.step("load_teacher"):
        teacher_ckpt = resolve_checkpoint("track_a_large", cfg)
        teacher = FrozenDeBERTaTeacher(
            base_model=cfg.model.track_a_large.base_model,
            num_labels=mcfg.num_labels,
            ckpt_path=teacher_ckpt,
        ).to(device)
    log.info(f"Teacher loaded from {teacher_ckpt} and frozen.")

    # ------------------------------------------------------------------
    # Student — initialise from supervised checkpoint if available,
    # otherwise start from scratch (03_track_b_99m.py must run first
    # to benefit from the warm start).
    # ------------------------------------------------------------------
    encoder_cfg = {
        "hidden_size":             mcfg.d_model,
        "num_layers":              mcfg.n_layers,
        "num_heads":               mcfg.n_heads,
        "intermediate_size":       mcfg.d_ff,
        "max_position_embeddings": seq_len,
        "dropout":                 mcfg.dropout,
        "pad_token_id":            student_tok.pad_token_id,
    }
    student = TrackB99MModel(
        vocab_size=student_tok.vocab_size,
        num_labels=mcfg.num_labels,
        encoder_cfg=encoder_cfg,
    ).to(device)

    try:
        supervised_ckpt = resolve_checkpoint(EXPERIMENT_TYPE, cfg)
        load_checkpoint(supervised_ckpt, student, device)
        log.info(f"Student warm-started from supervised checkpoint: {supervised_ckpt}")
    except FileNotFoundError:
        log.warning(
            "No supervised checkpoint found for track_b_99m. "
            "Starting distillation from random init. "
            "Run 03_track_b_99m.py first for best results."
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    epochs      = max(1, tcfg.max_steps // max(1, len(train_loader)))
    warmup      = int(tcfg.max_steps * tcfg.warmup_ratio)

    optimizer, scheduler = build_optimizer(
        student,
        lr=tcfg.peak_lr,
        weight_decay=tcfg.weight_decay,
        warmup_steps=warmup,
        total_steps=tcfg.max_steps,
        backbone_lr_multiplier=1.0,
    )

    tracker = CheckpointTracker(
        experiment_type=EXPERIMENT_TYPE,
        models_root=Path(cfg.paths.models),
        metric="macro_f1",
        mode="max",
        patience=5,
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    Path(cfg.mlflow.tracking_uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.project.name)
    with mlflow.start_run(run_name=f"{EXPERIMENT_TYPE}_distilled"):
        mlflow.log_params({
            "teacher":          cfg.model.track_a_large.base_model,
            "student":          EXPERIMENT_TYPE,
            "temperature":      temperature,
            "alpha_hard":       alpha,
            "alpha_soft":       tcfg.alpha_soft,
            "max_steps":        tcfg.max_steps,
            "batch_size":       tcfg.batch_size,
            "lr":               tcfg.peak_lr,
            "seq_len":          seq_len,
        })

        # Need a val loader without teacher IDs for standard evaluate()
        def _make_student_only_loader() -> DataLoader:
            from torch.utils.data import DataLoader as DL
            from train_utils import WafCollator

            class _BpeDs(Dataset):
                def __init__(self) -> None:
                    t = pq.read_table(splits / "val.parquet", columns=["raw", "label"])
                    self.texts  = t["raw"].to_pylist()
                    self.labels = t["label"].to_pylist()
                def __len__(self): return len(self.texts)
                def __getitem__(self, i):
                    ids = student_tok.encode(self.texts[i]).ids[:seq_len]
                    return {"input_ids": ids, "label": self.labels[i]}

            coll = WafCollator(student_tok, seq_len=seq_len)
            return DL(_BpeDs(), batch_size=tcfg.batch_size * 2, shuffle=False,
                      collate_fn=coll, num_workers=4)

        eval_loader = _make_student_only_loader()

        with timer.step("distillation"):
            for epoch in range(1, epochs + 1):
                train_m = run_distill_epoch(
                    student, teacher, train_loader,
                    optimizer, scheduler, device,
                    temperature=temperature, alpha=alpha,
                    accum_steps=tcfg.grad_accum_steps,
                )
                val_m = evaluate(student, eval_loader, device, num_labels=mcfg.num_labels)

                log.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"loss={train_m['loss']:.4f}  ce={train_m['ce_loss']:.4f}  "
                    f"kl={train_m['kl_loss']:.4f} | "
                    f"val_macro_f1={val_m['macro_f1']:.4f}"
                )

                mlflow.log_metrics(
                    flatten_metrics(train_m, "train_") |
                    flatten_metrics(val_m,   "val_"),
                    step=epoch,
                )

                tracker.step(
                    student, val_m,
                    meta={"epoch": epoch, "T": temperature, "alpha": alpha,
                          "train_kl": train_m["kl_loss"]},
                    step=epoch,
                )

                if tracker.should_stop:
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

        log.info(
            f"Distillation complete. Best macro_f1={tracker.best_value:.5f} "
            f"→ {tracker.best_ckpt}"
        )
        mlflow.log_artifact(str(tracker.best_ckpt / "checkpoint_meta.json"))
        timer.log_mlflow()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Knowledge distillation: DeBERTa → Track B 99M.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())