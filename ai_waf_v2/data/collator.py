"""
ai_waf_v2.data.collator
-------------------
DataCollator for the  encoder.

Handles dynamic padding within a batch (more efficient than padding
everything to seq_len at dataset level) and optionally mixes in
teacher soft logits for distillation training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WafCollator:
    """
    Collate a list of dataset items into a padded batch tensor dict.

    Parameters
    ----------
    pad_token_id : int
        Token ID used to pad input_ids.
    max_seq_len : int
        Hard cap: sequences are truncated to this length even if a
        batch item is longer (should not happen if the Dataset already
        truncates, but acts as a safety net).
    include_attack_class : bool
        If True, include the "attack_class" string list in the batch
        (useful during evaluation for per-class metric breakdown).
    """

    pad_token_id:          int  = 0
    max_seq_len:           int  = 256
    include_attack_class:  bool = False

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Parameters
        ----------
        batch : list[dict]
            Each element must contain:
                input_ids      : list[int]  (already truncated)
                attention_mask : list[int]
                labels         : int
            And optionally:
                teacher_logits : list[float]  (for distillation)
                attack_class   : str

        Returns
        -------
        dict with tensors ready for model forward().
        """
        # ── dynamic max length within this batch ──────
        max_len = min(
            self.max_seq_len,
            max(len(item["input_ids"]) for item in batch),
        )

        input_ids      = []
        attention_mask = []
        labels         = []
        teacher_logits = []
        has_teacher    = "teacher_logits" in batch[0]
        attack_classes = []

        for item in batch:
            ids  = item["input_ids"].tolist() if torch.is_tensor(item["input_ids"]) else item["input_ids"]
            mask = item["attention_mask"].tolist() if torch.is_tensor(item["attention_mask"]) else item["attention_mask"]

            # Truncate
            ids  = ids[:max_len]
            mask = mask[:max_len]

            # Pad
            pad_len = max_len - len(ids)
            ids  = ids  + [self.pad_token_id] * pad_len
            mask = mask + [0]                 * pad_len

            input_ids.append(ids)
            attention_mask.append(mask)

            lbl = item["labels"]
            labels.append(lbl.item() if torch.is_tensor(lbl) else lbl)

            if has_teacher:
                tl = item["teacher_logits"]
                teacher_logits.append(
                    tl.tolist() if torch.is_tensor(tl) else tl
                )

            if self.include_attack_class:
                attack_classes.append(item.get("attack_class", ""))

        out: dict[str, Any] = {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }

        if has_teacher:
            out["teacher_logits"] = torch.tensor(teacher_logits, dtype=torch.float)

        if self.include_attack_class:
            out["attack_class"] = attack_classes

        return out