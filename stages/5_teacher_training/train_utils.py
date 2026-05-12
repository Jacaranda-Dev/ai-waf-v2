"""
stages/5_training/train_utils.py
---------------------------------
Shared training primitives for all Stage 5 model scripts.

Centralising these avoids the infrastructure redundancy noted in the
critique — previously 01 and 02 duplicated collator/classifier logic,
and the runtime-injection pattern in 02 made MLflow run tracking
ambiguous.

Public API
----------
  WafClassifier   — pooling + MLP classification head (track-agnostic)
  WafCollator     — padding-aware batch builder for HF and BPE tokenizers
  build_optimizer — AdamW + linear-warmup scheduler
  run_epoch       — single training epoch with gradient accumulation
  evaluate        — full-dataset evaluation → metrics dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# 1. Classification head
# ---------------------------------------------------------------------------

class WafClassifier(nn.Module):
    """
    Track-agnostic classification head.

    Attaches a [CLS]-pooling + LayerNorm + dropout + linear projection
    on top of any encoder backbone.  Designed to accept backbone outputs
    from both HuggingFace AutoModel and the custom TinyTransformer used
    by Track B.

    Args:
        hidden_size:  Encoder hidden dimension.
        num_labels:   Number of output classes.
        dropout:      Dropout probability applied before the final linear.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, hidden) or (batch, hidden)
                            — accepts both; CLS pooling applied when 3-D.
        Returns:
            logits: (batch, num_labels)
        """
        if encoder_output.dim() == 3:
            # [CLS] token is always the first position
            x = encoder_output[:, 0, :]
        else:
            x = encoder_output
        return self.linear(self.drop(self.norm(x)))


# ---------------------------------------------------------------------------
# 2. Batch collator
# ---------------------------------------------------------------------------

@dataclass
class WafCollator:
    """
    Padding-aware batch builder compatible with HuggingFace tokenizers
    (Track A) and the custom HttpTokenizer (Track B).

    The distinction matters because HuggingFace tokenizers expose a
    `pad_token_id` attribute while HttpTokenizer may expose `pad_id`.
    This collator resolves either form transparently.

    Args:
        tokenizer:    Any tokenizer with an `encode` method.
        seq_len:      Maximum sequence length (truncates at this boundary).
        pad_id:       Override pad token ID; auto-resolved if None.
        label_dtype:  torch dtype for label tensors.
    """

    tokenizer:   Any
    seq_len:     int
    pad_id:      int | None = None
    label_dtype: torch.dtype = torch.long

    def __post_init__(self) -> None:
        if self.pad_id is None:
            # HuggingFace style
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                self.pad_id = self.tokenizer.pad_token_id
            # HttpTokenizer style
            elif hasattr(self.tokenizer, "pad_id"):
                self.pad_id = self.tokenizer.pad_id
            else:
                self.pad_id = 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dicts with keys 'input_ids' (list[int]) and 'label' (int).

        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels' as tensors.
        """
        ids_list: list[list[int]] = []
        labels:   list[int]       = []

        for item in batch:
            ids = item["input_ids"][: self.seq_len]
            ids_list.append(ids)
            labels.append(int(item["label"]))

        max_len = max(len(ids) for ids in ids_list)

        padded_ids = []
        masks      = []
        for ids in ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks,      dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=self.label_dtype),
        }


# ---------------------------------------------------------------------------
# 3. Optimizer + scheduler
# ---------------------------------------------------------------------------

def build_optimizer(
    model:          nn.Module,
    lr:             float = 2e-5,
    weight_decay:   float = 0.01,
    warmup_steps:   int   = 0,
    total_steps:    int   = 1,
    backbone_lr_multiplier: float = 1.0,
) -> tuple[AdamW, LambdaLR]:
    """
    AdamW with linear warmup + decay.

    Supports differential learning rates: classifier head uses `lr`,
    backbone uses `lr * backbone_lr_multiplier` (typically 0.1–0.5 for
    fine-tuning pre-trained weights).

    Returns:
        (optimizer, scheduler)
    """
    # Separate backbone params from classifier head params
    head_params     = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "linear" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr * backbone_lr_multiplier},
        {"params": head_params,     "lr": lr},
    ]

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 4. Training epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   AdamW,
    scheduler:   LambdaLR,
    device:      torch.device,
    accum_steps: int = 1,
    scaler:      torch.cuda.amp.GradScaler | None = None,
    loss_fn:     Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """
    Run one full training epoch with optional gradient accumulation and AMP.

    Args:
        model:       The complete model (backbone + classifier).
        loader:      DataLoader yielding collated batches.
        optimizer:   AdamW instance.
        scheduler:   LambdaLR instance (stepped every micro-batch).
        device:      Target device.
        accum_steps: Gradient accumulation window.
        scaler:      AMP GradScaler; pass None to disable mixed precision.
        loss_fn:     Callable(logits, labels) → scalar loss.
                     Defaults to cross-entropy.

    Returns:
        {"loss": float, "steps": int}
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0
    steps      = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attn_mask)
            loss   = loss_fn(logits, labels) / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            steps += 1

        total_loss += loss.item() * accum_steps

    return {"loss": total_loss / max(len(loader), 1), "steps": steps}


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    num_labels: int = 2,
) -> dict[str, float]:
    """
    Full-dataset evaluation.

    Returns:
        {
          "loss":      float,
          "accuracy":  float,
          "macro_f1":  float,   (requires sklearn)
          "per_class_f1": dict[int, float],
        }
    """
    from sklearn.metrics import f1_score

    model.eval()
    loss_fn    = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn_mask)
        total_loss += loss_fn(logits, labels).item()

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    n = len(loader)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    per_class   = f1_score(all_labels, all_preds, average=None,       zero_division=0)
    accuracy    = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)

    return {
        "loss":         total_loss / max(n, 1),
        "accuracy":     round(accuracy, 5),
        "macro_f1":     round(float(macro_f1), 5),
        "per_class_f1": {i: round(float(v), 5) for i, v in enumerate(per_class)},
    }