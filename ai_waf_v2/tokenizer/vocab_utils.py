"""
ai_waf_v2.tokenizer.vocab_utils
---------------------------
Vocabulary utilities for Track A: augmenting a pretrained BERT-base
WordPiece vocabulary with HTTP-specific and attack-pattern tokens.

The goal is to ensure common HTTP tokens (UNION, SELECT, script, ../)
are represented as single atomic units rather than split across
multiple subwords by the pretrained tokenizer.

Usage
-----
    from ai_waf_v2.tokenizer.vocab_utils import augment_pretrained_vocab

    tokenizer = augment_pretrained_vocab(
        base_model="bert-base-uncased",
        new_tokens=["UNION", "SELECT", "../", "%00", "onerror"],
        output_dir="tokenizers/track_a",
    )
"""

from __future__ import annotations

from pathlib import Path


def augment_pretrained_vocab(
    base_model:  str,
    new_tokens:  list[str],
    output_dir:  str | Path,
) -> object:
    """
    Load a pretrained HuggingFace tokenizer and inject new atomic tokens.

    Tokens that already exist in the vocabulary are skipped.
    The augmented tokenizer and its updated embedding table reference are
    returned so the caller can resize the model embedding layer.

    Parameters
    ----------
    base_model  : HuggingFace model identifier (e.g. "bert-base-uncased")
    new_tokens  : list of strings to add as atomic tokens
    output_dir  : directory to save the augmented tokenizer

    Returns
    -------
    tokenizer : PreTrainedTokenizerFast — augmented tokenizer
    n_added   : int — number of actually new tokens added
    """
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Filter out tokens already in the vocabulary
    existing = set(tokenizer.vocab.keys())
    to_add   = [t for t in new_tokens if t not in existing and t.lower() not in existing]

    n_added = tokenizer.add_tokens(to_add, special_tokens=False)

    tokenizer.save_pretrained(str(output_dir))

    return tokenizer, n_added


def measure_oov(
    tokenizer:    object,
    texts:        list[str],
    unk_token_id: int | None = None,
) -> dict[str, float]:
    """
    Measure Out-Of-Vocabulary rate and token fertility for a HuggingFace
    or custom tokenizer.

    Works with both HuggingFace PreTrainedTokenizer and the custom
    HttpTokenizer from ai_waf_v2.tokenizer.http_tokenizer.

    Parameters
    ----------
    tokenizer    : any tokenizer with an encode() method
    texts        : list of raw HTTP request strings
    unk_token_id : int | None — if None, tries tokenizer.unk_token_id

    Returns
    -------
    dict with: oov_rate, avg_seq_len, fertility
    """
    if unk_token_id is None:
        unk_token_id = getattr(tokenizer, "unk_token_id", 100)

    total_tokens = total_unk = total_chars = 0

    for text in texts:
        # Handle HuggingFace fast tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            enc = tokenizer.encode(text)
            ids = enc if isinstance(enc, list) else getattr(enc, "ids", enc)
        else:
            raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

        total_tokens += len(ids)
        total_unk    += sum(1 for i in ids if i == unk_token_id)
        total_chars  += len(text)

    n = max(len(texts), 1)
    return {
        "oov_rate":    round(total_unk / max(1, total_tokens), 6),
        "avg_seq_len": round(total_tokens / n, 2),
        "fertility":   round(total_tokens / max(1, total_chars), 4),
    }


def compare_tokenizers(
    tokenizer_a: object,
    tokenizer_b: object,
    texts:       list[str],
    labels:      list[str] | None = None,
) -> dict:
    """
    Compare two tokenizers on the same corpus, returning:
    - OOV rate per tokenizer
    - Average sequence length
    - Token fertility
    - Vocabulary overlap (if both expose get_vocab())
    - Per-attack-class OOV breakdown (if labels provided)

    Parameters
    ----------
    tokenizer_a : Track A tokenizer (pretrained + augmented)
    tokenizer_b : Track B tokenizer (custom BPE)
    texts       : list of raw HTTP request strings
    labels      : optional list of attack class labels (same length as texts)

    Returns
    -------
    dict with comparison metrics
    """
    stats_a = measure_oov(tokenizer_a, texts)
    stats_b = measure_oov(tokenizer_b, texts)

    result: dict = {
        "track_a": stats_a,
        "track_b": stats_b,
        "winner_oov":      "track_b" if stats_b["oov_rate"] < stats_a["oov_rate"] else "track_a",
        "winner_fertility": "track_b" if stats_b["fertility"] < stats_a["fertility"] else "track_a",
    }

    # Vocabulary overlap
    if hasattr(tokenizer_a, "vocab") and hasattr(tokenizer_b, "_tok"):
        vocab_a = set(tokenizer_a.vocab.keys())
        vocab_b = set(tokenizer_b._tok.get_vocab().keys())
        overlap = len(vocab_a & vocab_b) / max(1, len(vocab_a | vocab_b))
        result["vocab_jaccard_overlap"] = round(overlap, 4)

    # Per-class OOV breakdown
    if labels is not None:
        from collections import defaultdict

        class_texts: dict[str, list[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            class_texts[label].append(text)

        per_class_a, per_class_b = {}, {}
        for cls, cls_texts in class_texts.items():
            per_class_a[cls] = measure_oov(tokenizer_a, cls_texts)["oov_rate"]
            per_class_b[cls] = measure_oov(tokenizer_b, cls_texts)["oov_rate"]

        result["per_class_oov_track_a"] = per_class_a
        result["per_class_oov_track_b"] = per_class_b

    return result