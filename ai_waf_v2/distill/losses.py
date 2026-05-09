"""
ai_waf_v2distill.losses
--------------------
Loss functions for knowledge distillation.

The combined distillation loss is:

    L = α_hard × CE(student_logits, hard_labels)
      + α_soft × T² × KL(softmax(student_logits/T) ‖ softmax(teacher_logits/T))
      + w_mse  × MSE(projected_student_hidden, teacher_hidden)

where:
    T          = temperature (default 4.0 — softens teacher distribution)
    T²         = scale factor that normalises the KL term so it has
                 the same magnitude as CE regardless of T
    α_hard     = 0.3  (hard label weight)
    α_soft     = 0.7  (soft label weight)
    w_mse      = 0.0  (disabled when tokenizers differ between models)

Reference: Hinton et al. 2015 "Distilling the Knowledge in a Neural Network"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for WAF binary classification.

    Parameters
    ----------
    temperature  : float  — softens teacher/student logit distributions
    alpha_soft   : float  — weight on the KL (soft) loss term
    alpha_hard   : float  — weight on the CE (hard) loss term
    mse_weight   : float  — weight on the hidden-state MSE term (0 = disabled)
    label_smoothing : float — applied to the CE hard-label term

    Notes
    -----
    alpha_soft + alpha_hard should equal 1.0.
    mse_weight is independent and additive.
    """

    def __init__(
        self,
        temperature:     float = 4.0,
        alpha_soft:      float = 0.7,
        alpha_hard:      float = 0.3,
        mse_weight:      float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if abs(alpha_soft + alpha_hard - 1.0) > 1e-4:
            raise ValueError(
                f"alpha_soft ({alpha_soft}) + alpha_hard ({alpha_hard}) must sum to 1.0"
            )
        self.T               = temperature
        self.alpha_soft      = alpha_soft
        self.alpha_hard      = alpha_hard
        self.mse_weight      = mse_weight
        self.label_smoothing = label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # KLDivLoss expects log-probabilities as input, probabilities as target
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits:  torch.Tensor,           # (B, C)
        teacher_logits:  torch.Tensor,           # (B, C)
        hard_labels:     torch.Tensor,           # (B,)
        student_hidden:  torch.Tensor | None = None,  # (B, D_s)
        teacher_hidden:  torch.Tensor | None = None,  # (B, D_t)
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        student_logits  : raw logits from student model
        teacher_logits  : raw logits from teacher model (no gradient)
        hard_labels     : ground-truth integer labels
        student_hidden  : [CLS] hidden state from student (for MSE)
        teacher_hidden  : [CLS] hidden state from teacher (for MSE)

        Returns
        -------
        dict with keys:
            loss       : total combined loss (scalar, used for backward())
            loss_ce    : hard-label cross-entropy component
            loss_kl    : soft-label KL divergence component
            loss_mse   : hidden-state MSE component (0 if disabled)
        """
        # ── Hard label loss ──────────────────────────
        loss_ce = self.ce_loss(student_logits, hard_labels)

        # ── Soft label loss (KL divergence) ─────────
        # Temperature-scaled probabilities
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)

        student_log_probs = F.log_softmax(student_logits / self.T, dim=-1)

        # T² re-scales KL so its gradient magnitude matches CE
        loss_kl = self.kl_loss(student_log_probs, teacher_probs) * (self.T ** 2)

        # ── Hidden-state MSE (optional) ──────────────
        loss_mse = torch.tensor(0.0, device=student_logits.device)
        if (
            self.mse_weight > 0.0
            and student_hidden is not None
            and teacher_hidden is not None
        ):
            # Both must have been projected to the same dimension upstream
            loss_mse = F.mse_loss(student_hidden, teacher_hidden.detach())

        # ── Combined loss ────────────────────────────
        total = (
            self.alpha_hard * loss_ce
            + self.alpha_soft * loss_kl
            + self.mse_weight * loss_mse
        )

        return {
            "loss":     total,
            "loss_ce":  loss_ce.detach(),
            "loss_kl":  loss_kl.detach(),
            "loss_mse": loss_mse.detach(),
        }


class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy computed against soft (probability) targets rather
    than hard integer labels.  Useful for label-smoothed distillation
    without temperature scaling.

    Parameters
    ----------
    reduction : "mean" | "sum" | "none"
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C)
        targets: torch.Tensor,   # (B, C) — soft probability targets
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)      # (B, C)
        loss = -(targets * log_probs).sum(dim=-1)       # (B,)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss