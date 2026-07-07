"""
ai_waf_v2.models.student
--------------------
Student model for knowledge distillation.

The student shares the exact same WafClassifier architecture as the
teacher but with a much smaller configuration (default: 256d, 6 layers,
4 heads ≈ 10M parameters).

Key difference from the teacher: the student's ClassificationHead
projects from a smaller d_model, and the model exposes
``prepare_for_int8_quantization()`` for bitsandbytes QAT.

Usage
-----
    from ai_waf_v2.models.student import StudentClassifier
    from ai_waf_v2.utils.config import load_config

    cfg     = load_config()
    student = StudentClassifier.from_config(cfg.model.student)

    # During distillation training the student also accepts
    # teacher_hidden for the optional MSE alignment loss:
    out = student(input_ids, attention_mask, labels=labels,
                  teacher_hidden=teacher_hidden)
    # out["loss_ce"] + out["loss_mse"] available separately
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_waf_v2.models.encoder import WafEncoder
from ai_waf_v2.models.head import ClassificationHead

if TYPE_CHECKING:
    from ai_waf_v2.utils.config import StudentModelConfig


class StudentClassifier(nn.Module):
    """
    Smaller encoder classifier used as the distillation student.

    Adds an optional hidden-state projection layer (teacher_d_model →
    student d_model) for MSE alignment loss when teacher and student
    have different hidden dimensions.
    """

    def __init__(
        self,
        encoder:        WafEncoder,
        head:           ClassificationHead,
        teacher_d_model: int | None = None,  # if set, adds a projection for MSE loss
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head    = head

        student_d = encoder.d_model
        if teacher_d_model is not None and teacher_d_model != student_d:
            # Linear projection to align student hidden space with teacher's
            self.hidden_projection: nn.Linear | None = nn.Linear(
                student_d, teacher_d_model, bias=False
            )
        else:
            self.hidden_projection = None

    @classmethod
    def from_config(
        cls,
        cfg: StudentModelConfig,
        teacher_d_model: int | None = None,
    ) -> StudentClassifier:
        encoder = WafEncoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            max_seq_len=cfg.max_seq_len,
            pad_token_id=cfg.pad_token_id,
        )
        head = ClassificationHead(
            d_model=cfg.d_model,
            num_labels=cfg.num_labels,
            dropout=cfg.dropout,
        )
        return cls(encoder, head, teacher_d_model=teacher_d_model)

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        labels:           torch.Tensor | None = None,
        teacher_hidden:   torch.Tensor | None = None,
        label_smoothing:  float = 0.0,
        mse_weight:       float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids, attention_mask : standard encoder inputs
        labels          : (B,)  hard labels for CE loss
        teacher_hidden  : (B, D_teacher) — used for MSE alignment loss
        label_smoothing : float
        mse_weight      : weight for the hidden-state MSE loss term

        Returns
        -------
        dict keys: logits, hidden, loss_ce (if labels), loss_mse (if teacher_hidden)
        """
        cls_hidden = self.encoder(input_ids, attention_mask)   # (B, D_student)
        logits     = self.head(cls_hidden)                     # (B, C)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden": cls_hidden,
        }

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            out["loss_ce"] = loss_fn(logits, labels)

        if teacher_hidden is not None and mse_weight > 0.0:
            projected = (
                self.hidden_projection(cls_hidden)
                if self.hidden_projection is not None
                else cls_hidden
            )
            out["loss_mse"] = F.mse_loss(projected, teacher_hidden.detach())
        else:
            out["loss_mse"] = torch.tensor(0.0, device=logits.device)

        return out

    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        threshold:      float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out   = self.forward(input_ids, attention_mask)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            preds = (probs >= threshold).long()
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def prepare_for_int8_quantization(
        self,
        exclude_modules: list[str] | None = None,
    ) -> StudentClassifier:
        """
        Replace Linear layers with bitsandbytes Linear8bitLt for INT8
        quantization-aware training.

        Parameters
        ----------
        exclude_modules : list[str] | None
            Names of top-level child modules to skip.  Defaults to ``["head"]``
            because the 2-class output projection has dimensions too small for
            the cublasLt INT8 kernel (minimum alignment requirements).

        Requires: pip install bitsandbytes
        Call BEFORE moving model to GPU.
        """
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for INT8 quantization. "
                "Install with: pip install bitsandbytes"
            ) from e

        excluded = set(exclude_modules if exclude_modules is not None else ["head"])

        def _replace_linear(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    new = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
                    new.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        new.bias.data = child.bias.data.clone()
                    setattr(module, name, new)
                else:
                    _replace_linear(child)

        for name, child in self.named_children():
            if name not in excluded:
                _replace_linear(child)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: StudentModelConfig,
        teacher_d_model: int | None = None,
        map_location: str = "cpu",
    ) -> StudentClassifier:
        model = cls.from_config(cfg, teacher_d_model=teacher_d_model)
        # weights_only=False: bnb state dicts contain plain Python strings
        # (weight_format) that weights_only=True rejects.
        state = torch.load(path, map_location="cpu", weights_only=False)

        if any(k.endswith(".SCB") for k in state):
            # Checkpoint was saved after INT8 quantization.
            # Dequantize back to fp32: w_fp32 = w_int8 * (SCB / 127).
            # This avoids cublasLt kernel failures on tiny output dims (e.g. the
            # 2-class head) and makes the checkpoint portable to CPU inference.
            clean: dict[str, torch.Tensor] = {}
            for k, v in state.items():
                if k.endswith((".SCB", ".weight_format")):
                    continue
                if k.endswith(".weight") and v.dtype == torch.int8:
                    scb_key = k[: -len("weight")] + "SCB"
                    if scb_key in state:
                        scb = state[scb_key].float()          # (out_features,)
                        clean[k] = v.float() * scb.view(-1, 1) / 127.0
                        continue
                clean[k] = v
            model.load_state_dict(clean)
        else:
            model.load_state_dict(state)

        return model.to(map_location)