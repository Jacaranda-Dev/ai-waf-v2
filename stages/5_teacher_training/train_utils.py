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

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

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

_NAN_STREAK_LIMIT = 20  # consecutive non-finite losses before aborting


def run_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   AdamW,
    scheduler:   LambdaLR,
    device:      torch.device,
    accum_steps: int = 1,
    scaler:      Any | None = None,
    loss_fn:     Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """
    Run one full training epoch with optional gradient accumulation and AMP.

    When `scaler` is provided (a torch.cuda.amp.GradScaler), uses fp16 autocast
    with dynamic loss scaling — required for DeBERTa-v3 which is numerically
    unstable under bf16.  Without a scaler, falls back to bf16.

    Args:
        model:       The complete model (backbone + classifier).
        loader:      DataLoader yielding collated batches.
        optimizer:   AdamW instance.
        scheduler:   LambdaLR instance (stepped after each accumulation window).
        device:      Target device.
        accum_steps: Gradient accumulation window.
        scaler:      GradScaler for fp16 training; None → bf16.
        loss_fn:     Callable(logits, labels) → scalar loss.
                     Defaults to cross-entropy.

    Returns:
        {"loss": float, "steps": int}
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if device.type == "cuda":
        dtype = torch.float16 if scaler is not None else torch.bfloat16
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    model.train()
    total_loss = 0.0
    steps      = 0
    nan_streak = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="train", unit="batch", dynamic_ncols=True, leave=False)
    for i, batch in enumerate(pbar):
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        with autocast_ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask)
            loss   = loss_fn(logits, labels) / accum_steps

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            nan_streak += 1
            pbar.write(f"[warn] non-finite loss at batch {i}; skipping backward")
            if nan_streak >= _NAN_STREAK_LIMIT:
                raise RuntimeError(
                    f"Training diverged: {nan_streak} consecutive non-finite losses. "
                    "Model weights are likely NaN — check LR, mixed-precision dtype, "
                    "and data quality. For DeBERTa pass a GradScaler to use fp16."
                )
            optimizer.zero_grad()
            continue

        nan_streak = 0

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(grad_norm.item()):
                pbar.write(f"[warn] non-finite grad_norm={grad_norm.item():.4f} at step {steps}; skipping optimizer step")
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1

        total_loss += loss_val * accum_steps
        pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

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

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    for batch in tqdm(loader, desc="eval", unit="batch", dynamic_ncols=True, leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        with autocast_ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask)
        total_loss += loss_fn(logits.float(), labels).item()

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


def flatten_metrics(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a metrics dict for MLflow — expands nested dicts into dotted keys."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                out[f"{prefix}{k}_{sub_k}"] = float(sub_v)
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out