"""
stages/5_teacher_training/track_b_model.py
-------------------------------------------
Shared Track B model classes imported by both 03_track_b_99m.py and
03b_distill_track_b.py.  Extracted here because Python cannot import
modules whose names begin with a digit.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from train_utils import WafClassifier


class TinyTransformerEncoder(nn.Module):
    """
    Compact BERT-style encoder with custom vocab embedding.

    Initialised from scratch against the Track B vocabulary — no
    pre-trained weights are loaded here.  Knowledge distillation in
    03b_distill_track_b.py subsequently transfers teacher signal.

    Args:
        vocab_size:              Size of the Track B BPE vocabulary.
        hidden_size:             Embedding + transformer hidden dimension.
        num_layers:              Number of transformer encoder layers.
        num_heads:               Number of attention heads.
        intermediate_size:       FFN intermediate dimension.
        max_position_embeddings: Maximum sequence length.
        dropout:                 Dropout probability.
        pad_token_id:            Padding token ID (masked in attention).
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

        self.word_embeddings  = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embeddings   = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embed = nn.Embedding(2, hidden_size)
        self.embed_norm       = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embed_drop       = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability from random init
        )
        # enable_nested_tensor=False: nested-tensor optimisation is incompatible
        # with norm_first=True; suppress the UserWarning it would otherwise emit.
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
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
        input_ids:      torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, L, H)
        B, L = input_ids.shape
        device = input_ids.device

        positions   = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        token_types = torch.zeros_like(input_ids)

        x = (
            self.word_embeddings(input_ids)
            + self.pos_embeddings(positions)
            + self.token_type_embed(token_types)
        )
        x = self.embed_drop(self.embed_norm(x))

        # src_key_padding_mask: True where tokens should be ignored
        pad_mask = attention_mask == 0
        return self.encoder(x, src_key_padding_mask=pad_mask)


class TrackB99MModel(nn.Module):
    """TinyTransformerEncoder + WafClassifier head for Track B."""

    def __init__(
        self,
        vocab_size:  int,
        num_labels:  int,
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
        """Return [CLS] embedding without the classification head — used by distillation."""
        hidden = self.encoder(input_ids, attention_mask)
        return hidden[:, 0, :]  # (B, H)
