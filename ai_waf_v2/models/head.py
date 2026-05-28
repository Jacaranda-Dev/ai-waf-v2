"""
ai_waf_v2.models.head
-----------------
Classification head for the WAF encoder.

WafClassifier   — full model: WafEncoder + ClassificationHead
ClassificationHead — the linear head alone (used in distillation)

Usage
-----
    from ai_waf_v2.models.head import WafClassifier
    from ai_waf_v2.utils.config import load_config

    cfg = load_config()
    arch = cfg.model.track_b_99m
    model = WafClassifier.from_config(arch)
    logits = model(input_ids, attention_mask)   # (B, num_classes)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ai_waf_v2.models.encoder import WafEncoder

if TYPE_CHECKING:
    from ai_waf_v2.utils.config import ModelArchConfig


class ClassificationHead(nn.Module):
    """
    Single linear layer that maps [CLS] hidden state → class logits.

    A small Dropout before the linear layer acts as a regulariser
    during fine-tuning (matches BERT/DeBERTa convention).

    Parameters
    ----------
    d_model    : int   — input dimension (must match encoder output)
    num_labels : int   — number of output classes (2 for binary WAF)
    dropout    : float — dropout applied to CLS before linear
    """

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(d_model, num_labels)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_hidden : (B, D)

        Returns
        -------
        logits : (B, num_labels)  — raw unNormalized scores
        """
        return self.linear(self.dropout(cls_hidden))


class WafClassifier(nn.Module):
    """
    Full classification model: WafEncoder → ClassificationHead.

    This is the primary training artefact for Track B.

    Parameters
    ----------
    encoder : WafEncoder
    head    : ClassificationHead
    """

    def __init__(self, encoder: WafEncoder, head: ClassificationHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head    = head

    @classmethod
    def from_config(cls, arch: "ModelArchConfig") -> "WafClassifier":
        """Instantiate from a ModelArchConfig (from pipeline.yaml)."""
        encoder = WafEncoder(
            vocab_size=arch.vocab_size,
            d_model=arch.d_model,
            n_layers=arch.n_layers,
            n_heads=arch.n_heads,
            d_ff=arch.d_ff,
            dropout=arch.dropout,
            attn_dropout=arch.attn_dropout,
            max_seq_len=arch.max_seq_len,
            pad_token_id=arch.pad_token_id,
        )
        head = ClassificationHead(
            d_model=arch.d_model,
            num_labels=arch.num_labels,
            dropout=arch.dropout,
        )
        return cls(encoder, head)

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
        labels:         torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids, attention_mask : standard encoder inputs
        labels : (B,) long — if provided, loss is computed and returned
        label_smoothing : float — applied to CrossEntropyLoss when labels given

        Returns
        -------
        dict with keys:
            "logits" : (B, num_labels)
            "loss"   : scalar tensor (only if labels provided)
            "hidden" : (B, D) — CLS hidden state (used in distillation)
        """
        cls_hidden = self.encoder(input_ids, attention_mask)   # (B, D)
        logits     = self.head(cls_hidden)                     # (B, C)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden": cls_hidden,
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            out["loss"] = loss_fn(logits, labels)

        return out

    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        threshold:      float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference: returns predicted class and
        confidence (probability of the positive / malicious class).

        Parameters
        ----------
        threshold : float
            Probability threshold for the malicious class (class index 1).

        Returns
        -------
        preds : (B,) long  — 0=benign, 1=malicious
        probs : (B,)  float — P(malicious)
        """
        with torch.no_grad():
            out   = self.forward(input_ids, attention_mask)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            preds = (probs >= threshold).long()
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        arch: "ModelArchConfig",
        map_location: str = "cpu",
    ) -> "WafClassifier":
        model = cls.from_config(arch)
        state = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model