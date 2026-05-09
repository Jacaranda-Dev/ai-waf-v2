"""
ai_waf_v2.models.encoder
--------------------
Transformer encoder built from scratch for WAF request classification.

Architecture (default — Track B 99M):
    Embedding       : vocab_size=8000, d_model=768, max_seq=256
    Encoder layers  : 13 × (MultiHeadAttention + FFN + LayerNorm)
    Pooling         : [CLS] token (position 0)
    Head            : Linear(768 → num_classes)  ← in ai_waf_v2.models.head

The class is self-contained: it does NOT depend on HuggingFace
transformers so that architecture hyperparameters are fully ours.

Flash Attention 2
-----------------
When torch >= 2.2 and a CUDA device is available, attention uses
``torch.nn.functional.scaled_dot_product_attention`` which dispatches
to Flash Attention 2 on Ampere+ GPUs (RTX 4000 Ada, RTX 5090) with
no code changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Position embedding (learned, standard BERT-style)
# ─────────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        return self.embedding(positions)                                # (1, T, D)


# ─────────────────────────────────────────────────────────
# Multi-head self-attention
# ─────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional causal mask.

    Uses ``F.scaled_dot_product_attention`` (PyTorch 2.0+) which
    dispatches to Flash Attention 2 on supported hardware automatically.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = math.sqrt(self.head_dim)  # kept for reference; SDPA handles it

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = attn_dropout

    def forward(
        self,
        hidden: torch.Tensor,           # (B, T, D)
        attention_mask: torch.Tensor,   # (B, T)  — 1=attend, 0=pad
    ) -> torch.Tensor:
        B, T, D = hidden.shape
        H, Dh   = self.n_heads, self.head_dim

        def _proj_split(proj: nn.Linear) -> torch.Tensor:
            # (B, T, D) → (B, H, T, Dh)
            return proj(hidden).view(B, T, H, Dh).transpose(1, 2)

        Q = _proj_split(self.q_proj)
        K = _proj_split(self.k_proj)
        V = _proj_split(self.v_proj)

        # Convert padding mask to additive attention bias
        # attention_mask: (B, T) → (B, 1, 1, T)  (broadcast over heads and query pos)
        # padded positions → -inf so softmax ignores them
        key_padding_mask = attention_mask[:, None, None, :]          # (B, 1, 1, T)
        attn_bias = torch.zeros(B, H, T, T, dtype=Q.dtype, device=Q.device)
        attn_bias = attn_bias.masked_fill(key_padding_mask == 0, float("-inf"))

        # Flash Attention 2 path (PyTorch ≥ 2.0)
        # dropout_p is only applied during training
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
        )  # (B, H, T, Dh)

        # Merge heads: (B, H, T, Dh) → (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_out)


# ─────────────────────────────────────────────────────────
# Feed-forward block (GELU, 4× expansion)
# ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────
# Single encoder layer
# ─────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Pre-LayerNorm encoder layer (more stable than post-LN for deep models).

    Pre-LN:   x → LayerNorm → Attention → + x → LayerNorm → FFN → + x
    Post-LN:  x → Attention → + x → LayerNorm → FFN → + x → LayerNorm
    """

    def __init__(
        self,
        d_model:      int,
        n_heads:      int,
        d_ff:         int,
        dropout:      float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # (B, T, D)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        # Self-attention sub-layer (pre-LN)
        x = x + self.drop(self.attn(self.norm1(x), attention_mask))
        # FFN sub-layer (pre-LN)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────
# Full encoder (teacher — 99M params)
# ─────────────────────────────────────────────────────────

class WafEncoder(nn.Module):
    """
    Transformer encoder for WAF binary classification.

    The model appends a [CLS] token at position 0 (via the embedding
    table; the tokenizer must produce token_id=cls_token_id at position 0).
    The [CLS] hidden state from the last layer is used as the sequence
    representation and fed to the classification head.

    Parameters
    ----------
    vocab_size   : int   — must match tokenizer vocab size
    d_model      : int   — hidden dimension (default 768)
    n_layers     : int   — number of encoder layers (default 13)
    n_heads      : int   — attention heads (default 12)
    d_ff         : int   — FFN inner dimension (default 4 × d_model = 3072)
    dropout      : float — residual + FFN dropout rate
    attn_dropout : float — attention weight dropout rate
    max_seq_len  : int   — maximum sequence length (default 256)
    pad_token_id : int   — token ID that is masked out in attention
    """

    def __init__(
        self,
        vocab_size:   int   = 8000,
        d_model:      int   = 768,
        n_layers:     int   = 13,
        n_heads:      int   = 12,
        d_ff:         int   = 3072,
        dropout:      float = 0.1,
        attn_dropout: float = 0.1,
        max_seq_len:  int   = 256,
        pad_token_id: int   = 0,
    ) -> None:
        super().__init__()
        self.d_model      = d_model
        self.pad_token_id = pad_token_id

        # ── Embeddings ───────────────────────────────
        self.token_embedding    = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.embed_dropout      = nn.Dropout(dropout)
        self.embed_norm         = nn.LayerNorm(d_model)

        # ── Encoder stack ────────────────────────────
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_layers)
        ])

        # ── Final layer norm (post-stack, pre-LN style) ──
        self.final_norm = nn.LayerNorm(d_model)

        # ── Weight initialisation ────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """BERT-style weight initialisation: N(0, 0.02) for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,  # (B, T)
        attention_mask: torch.Tensor,  # (B, T)  — 1=real token, 0=pad
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : (B, T) long
        attention_mask : (B, T) long

        Returns
        -------
        cls_hidden : (B, D) float  — [CLS] token representation from final layer
        """
        B, T = input_ids.shape

        tok_emb = self.token_embedding(input_ids)                     # (B, T, D)
        pos_emb = self.position_embedding(T, input_ids.device)        # (1, T, D)
        x = self.embed_norm(self.embed_dropout(tok_emb + pos_emb))   # (B, T, D)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.final_norm(x)   # (B, T, D)

        # Pool: take the [CLS] token at position 0
        cls_hidden = x[:, 0, :]  # (B, D)
        return cls_hidden

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Return parameter counts broken down by component."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        emb = count(self.token_embedding) + count(self.position_embedding)
        attn_total = sum(
            count(layer.attn) for layer in self.layers
        )
        ffn_total = sum(
            count(layer.ffn) for layer in self.layers
        )
        ln_total = sum(
            count(layer.norm1) + count(layer.norm2) for layer in self.layers
        ) + count(self.embed_norm) + count(self.final_norm)

        return {
            "embeddings":   emb,
            "attention":    attn_total,
            "ffn":          ffn_total,
            "layer_norms":  ln_total,
            "total":        self.count_parameters(),
        }