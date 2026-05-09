"""
ai_waf_v2.distill.trainer
---------------------
Distillation training loop: teacher (frozen 99M) → student (INT8 ~10M).

Usage
-----
    from ai_waf_v2.distill.trainer import DistillationTrainer
    from ai_waf_v2.utils.config import load_config

    cfg     = load_config()
    trainer = DistillationTrainer(cfg, teacher, student, train_loader, val_loader)
    trainer.train()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ai_waf_v2.distill.losses import DistillationLoss
from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from ai_waf_v2.models.head import WafClassifier
    from ai_waf_v2.models.student import StudentClassifier
    from ai_waf_v2.utils.config import PipelineConfig

log = get_logger(__name__)


class DistillationTrainer:
    """
    Full distillation training loop with:
    - Teacher frozen in eval mode
    - AdamW with weight-decay split (no decay on bias/LN)
    - Linear warmup + cosine decay LR schedule
    - bf16 mixed precision (RTX 4000 Ada / RTX 5090 native)
    - Gradient accumulation
    - Gradient clipping
    - MLflow metric logging
    - Best-checkpoint saving on val AUC-PR

    Parameters
    ----------
    cfg          : PipelineConfig
    teacher      : WafClassifier — frozen, already on device
    student      : StudentClassifier — trained in this loop
    train_loader : DataLoader yielding batches with input_ids, attention_mask,
                   labels, and optionally teacher_logits (pre-computed)
    val_loader   : DataLoader for validation (no teacher needed)
    device       : torch.device (defaults to CUDA if available)
    """

    def __init__(
        self,
        cfg:          "PipelineConfig",
        teacher:      "WafClassifier",
        student:      "StudentClassifier",
        train_loader: "DataLoader",
        val_loader:   "DataLoader",
        device:       torch.device | None = None,
    ) -> None:
        self.cfg          = cfg
        self.dcfg         = cfg.training.distillation
        self.teacher      = teacher
        self.student      = student
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student.to(self.device)

        # ── Loss ──────────────────────────────────────
        self.loss_fn = DistillationLoss(
            temperature=self.dcfg.temperature,
            alpha_soft=self.dcfg.alpha_soft,
            alpha_hard=self.dcfg.alpha_hard,
            mse_weight=self.dcfg.hidden_mse_weight,
        )

        # ── Optimiser (weight decay split) ────────────
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        params = [
            {
                "params": [
                    p for n, p in student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.dcfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(params, lr=self.dcfg.peak_lr)

        # ── LR schedule: linear warmup + cosine decay ─
        warmup_steps  = int(self.dcfg.max_steps * self.dcfg.warmup_ratio)
        self.scheduler = _WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.dcfg.max_steps,
        )

        # ── Mixed precision ────────────────────────────
        self._use_bf16 = (
            self.dcfg.precision == "bf16"
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self._autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self._use_bf16
            else torch.amp.autocast("cuda", enabled=False)
        )

        self._best_auc_pr  = 0.0
        self._global_step  = 0
        self._output_dir   = Path(cfg.model.student.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────
    def train(self) -> None:
        """Run the full distillation training loop."""
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment

        init_experiment(self.cfg)

        log.info(
            f"Distillation: teacher={self.teacher.count_parameters():,} params | "
            f"student={self.student.count_parameters():,} params | "
            f"device={self.device} | bf16={self._use_bf16}"
        )

        self.student.train()
        self.optimizer.zero_grad()

        step_iter  = iter(self.train_loader)
        step       = 0

        while step < self.dcfg.max_steps:
            try:
                batch = next(step_iter)
            except StopIteration:
                step_iter = iter(self.train_loader)
                batch     = next(step_iter)

            loss_dict = self._train_step(batch, step)

            # Gradient accumulation: only update every N micro-steps
            if (step + 1) % self.dcfg.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.dcfg.grad_clip_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self._global_step += 1

                if self._global_step % 100 == 0:
                    log.info(
                        f"step={self._global_step:>6} | "
                        f"loss={loss_dict['loss']:.4f} | "
                        f"kl={loss_dict['loss_kl']:.4f} | "
                        f"ce={loss_dict['loss_ce']:.4f} | "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    mlflow.log_metrics(
                        {
                            "train/loss":     loss_dict["loss"],
                            "train/loss_kl":  loss_dict["loss_kl"],
                            "train/loss_ce":  loss_dict["loss_ce"],
                            "train/lr":       self.scheduler.get_last_lr()[0],
                        },
                        step=self._global_step,
                    )

                # Validation
                val_every = max(1, self.dcfg.max_steps // (self.dcfg.grad_accum_steps * 20))
                if self._global_step % val_every == 0:
                    val_metrics = self._validate()
                    mlflow.log_metrics(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=self._global_step,
                    )
                    auc_pr = val_metrics.get("auc_pr", 0.0)
                    if auc_pr > self._best_auc_pr:
                        self._best_auc_pr = auc_pr
                        self.student.save(self._output_dir / "best_student.pt")
                        log.info(f"  ✓ new best AUC-PR={auc_pr:.4f} — checkpoint saved")

            step += 1

        # Save final weights
        self.student.save(self._output_dir / "final_student.pt")
        log.info(f"Training complete. Best val AUC-PR: {self._best_auc_pr:.4f}")

    # ─────────────────────────────────────────────────
    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, float]:
        """One micro-step (before gradient accumulation update)."""
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)

        with self._autocast_ctx:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out    = self.teacher(input_ids, attention_mask)
                teacher_logits = teacher_out["logits"]
                teacher_hidden = teacher_out["hidden"]

            # Student forward
            student_out = self.student(
                input_ids,
                attention_mask,
                labels=labels,
                teacher_hidden=teacher_hidden if self.dcfg.hidden_mse_weight > 0 else None,
                mse_weight=self.dcfg.hidden_mse_weight,
            )

            loss_dict = self.loss_fn(
                student_logits=student_out["logits"],
                teacher_logits=teacher_logits,
                hard_labels=labels,
                student_hidden=student_out.get("hidden"),
                teacher_hidden=teacher_hidden,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss_dict["loss"] / self.dcfg.grad_accum_steps

        scaled_loss.backward()

        return {k: v.item() for k, v in loss_dict.items()}

    def _validate(self) -> dict[str, float]:
        """Run full validation pass and return metrics dict."""
        self.student.eval()
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                with self._autocast_ctx:
                    preds, probs = self.student.predict(input_ids, attention_mask)

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        preds_t  = torch.cat(all_preds)
        probs_t  = torch.cat(all_probs)
        labels_t = torch.cat(all_labels)

        metrics = compute_metrics(
            preds=preds_t,
            probs=probs_t,
            labels=labels_t,
        )
        self.student.train()
        return metrics


# ─────────────────────────────────────────────────────────
# LR scheduler: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────

class _WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_steps:  int,
        total_steps:   int,
        min_lr_ratio:  float = 0.1,
    ) -> None:
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        super().__init__(optimizer, lr_lambda)