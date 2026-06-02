"""
stages/4_tokenization/tokenizer_eval.py
----------------------------------------
Shared evaluation utilities for Track A / Track B tokenizer comparison.

Addresses:
  - Sequential sampling bias  → stratified_sample()
  - Truncation blindness      → compute_truncation_rate()
  - Logic redundancy          → compute_full_metrics() / build_unified_report()
  - Token shadowing           → check_token_shadowing()
  - Subword compression       → subword_char_ratio()

Import this module instead of duplicating logic across scripts 02, 04, 05.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Protocol, Sequence

# ---------------------------------------------------------------------------
# Tokenizer duck-type protocol
# Accepts both HuggingFace AutoTokenizer and HttpTokenizer.
# ---------------------------------------------------------------------------

class _Tokenizer(Protocol):
    def encode(self, text: str, **kwargs) -> list[int]: ...
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...


# ---------------------------------------------------------------------------
# 1. Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    texts: list[str],
    labels: list[str],
    n: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Return a stratified random sample of size *n* that preserves
    the class distribution of *labels*.

    Fixes the sequential-slice bias in the original [:5000] calls,
    which could silently exclude minority attack classes when the
    Parquet files are ordered by timestamp or category.

    Args:
        texts:  Raw HTTP request strings.
        labels: Corresponding attack_class labels.
        n:      Target sample size.
        seed:   RNG seed for reproducibility.

    Returns:
        (sampled_texts, sampled_labels) as parallel lists.
    """
    if len(texts) != len(labels):
        raise ValueError(f"texts and labels must be the same length ({len(texts)} vs {len(labels)})")

    rng = random.Random(seed)

    # Group indices by class
    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    classes = sorted(class_indices.keys())
    n_classes = len(classes)
    n_per_class = max(1, n // n_classes)

    chosen: list[int] = []
    for cls in classes:
        pool = class_indices[cls]
        k = min(n_per_class, len(pool))
        chosen.extend(rng.sample(pool, k))

    # Top-up to exactly n if rounding left us short
    if len(chosen) < n:
        unchosen = list(set(range(len(texts))) - set(chosen))
        extra_k = min(n - len(chosen), len(unchosen))
        chosen.extend(rng.sample(unchosen, extra_k))

    rng.shuffle(chosen)
    chosen = chosen[:n]

    return [texts[i] for i in chosen], [labels[i] for i in chosen]


# ---------------------------------------------------------------------------
# 2. Core metric primitives
# ---------------------------------------------------------------------------

def _tokenize_batch(
    tokenizer: _Tokenizer,
    texts: list[str],
    seq_len: int | None = None,
    add_special_tokens: bool = True,
) -> list[list[str]]:
    """Encode each text and return lists of subword token strings."""
    results: list[list[str]] = []
    for text in texts:
        try:
            # HuggingFace transformers path: encode() returns list[int]
            ids = tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,          # Intentionally NOT truncating here
            )
            tokens = tokenizer.convert_ids_to_tokens(ids)
        except TypeError:
            # HttpTokenizer.encode() does not accept keyword arguments and
            # returns an Encoding object, not list[int]. Use the dedicated
            # no-truncation path so truncation_rate is computed correctly.
            if hasattr(tokenizer, "encode_no_truncation"):
                enc = tokenizer.encode_no_truncation(text)
            else:
                enc = tokenizer.encode(text)
            tokens = enc.tokens
        results.append(tokens)
    return results


def compute_oov_rate(token_seqs: list[list[str]], unk_token: str = "[UNK]") -> float:
    """Fraction of subword tokens that are [UNK]."""
    total = unk_count = 0
    for seq in token_seqs:
        total += len(seq)
        unk_count += sum(1 for t in seq if t == unk_token)
    return round(unk_count / max(total, 1), 6)


def compute_avg_seq_len(token_seqs: list[list[str]]) -> float:
    """Mean number of subword tokens per request."""
    if not token_seqs:
        return 0.0
    return round(sum(len(s) for s in token_seqs) / len(token_seqs), 4)


def compute_fertility(token_seqs: list[list[str]], texts: list[str]) -> float:
    """
    Fertility = total subword tokens / total characters.
    High fertility → excessive fragmentation, more truncation risk.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(s) for s in token_seqs)
    return round(total_tokens / max(total_chars, 1), 6)


def compute_truncation_rate(token_seqs: list[list[str]], seq_len: int) -> float:
    """
    Fraction of requests whose token count EXCEEDS seq_len.

    Addresses the truncation-blindness gap: a high-fertility tokenizer
    may silently drop the tail of long requests, where many XSS / file-
    upload payloads reside.
    """
    truncated = sum(1 for s in token_seqs if len(s) > seq_len)
    return round(truncated / max(len(token_seqs), 1), 6)


def subword_char_ratio(token_seqs: list[list[str]], texts: list[str]) -> float:
    """
    Mean characters-per-subword token (inverse of fertility).
    Higher → more compression; lower → noisier vocabulary.
    """
    total_tokens = sum(len(s) for s in token_seqs)
    total_chars = sum(len(t) for t in texts)
    return round(total_chars / max(total_tokens, 1), 4)


# ---------------------------------------------------------------------------
# 3. Per-class metric breakdown
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    tokenizer: _Tokenizer,
    texts: list[str],
    labels: list[str],
    seq_len: int,
    unk_token: str = "[UNK]",
) -> dict[str, dict[str, float]]:
    """
    Per-attack-class OOV rate, fertility, and truncation rate.

    Returns a dict keyed by class name, e.g.:
      {
        "sqli":           {"oov_rate": 0.012, "fertility": 0.73, "truncation_rate": 0.04},
        "path_traversal": {"oov_rate": 0.031, ...},
        ...
      }
    """
    class_texts: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(texts, labels):
        class_texts[label].append(text)

    per_class: dict[str, dict[str, float]] = {}
    for cls, cls_texts in sorted(class_texts.items()):
        seqs = _tokenize_batch(tokenizer, cls_texts)
        per_class[cls] = {
            "oov_rate":        compute_oov_rate(seqs, unk_token),
            "fertility":       compute_fertility(seqs, cls_texts),
            "truncation_rate": compute_truncation_rate(seqs, seq_len),
            "n_samples":       len(cls_texts),
        }
    return per_class


# ---------------------------------------------------------------------------
# 4. Unified metric schema
# ---------------------------------------------------------------------------

def _compute_per_class_from_seqs(
    token_seqs: list[list[str]],
    texts: list[str],
    labels: list[str],
    seq_len: int,
    unk_token: str = "[UNK]",
) -> dict[str, dict[str, float]]:
    """Per-class metrics reusing already-computed token sequences (no re-tokenization)."""
    class_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for seq, text, label in zip(token_seqs, texts, labels):
        class_data[label][0].append(seq)
        class_data[label][1].append(text)

    per_class: dict[str, dict[str, float]] = {}
    for cls in sorted(class_data.keys()):
        seqs, cls_texts = class_data[cls]
        per_class[cls] = {
            "oov_rate":        compute_oov_rate(seqs, unk_token),
            "fertility":       compute_fertility(seqs, cls_texts),
            "truncation_rate": compute_truncation_rate(seqs, seq_len),
            "n_samples":       len(cls_texts),
        }
    return per_class


def compute_full_metrics(
    tokenizer: _Tokenizer,
    texts: list[str],
    labels: list[str],
    seq_len: int,
    unk_token: str = "[UNK]",
    track_name: str = "unknown",
) -> dict[str, Any]:
    """
    Compute the full standardised metric schema for one tokenizer track.

    Schema:
      {
        "track":             str,
        "n_samples":         int,
        "seq_len":           int,
        "oov_rate":          float,
        "avg_seq_len":       float,
        "fertility":         float,
        "truncation_rate":   float,
        "subword_char_ratio":float,
        "per_class":         { class_name: {oov_rate, fertility, truncation_rate, n_samples} }
      }
    """
    token_seqs = _tokenize_batch(tokenizer, texts)

    return {
        "track":              track_name,
        "n_samples":          len(texts),
        "seq_len":            seq_len,
        "oov_rate":           compute_oov_rate(token_seqs, unk_token),
        "avg_seq_len":        compute_avg_seq_len(token_seqs),
        "fertility":          compute_fertility(token_seqs, texts),
        "truncation_rate":    compute_truncation_rate(token_seqs, seq_len),
        "subword_char_ratio": subword_char_ratio(token_seqs, texts),
        "per_class":          _compute_per_class_from_seqs(
                                  token_seqs, texts, labels, seq_len, unk_token
                              ),
    }


# ---------------------------------------------------------------------------
# 5. Token shadowing analysis (Track A specific)
# ---------------------------------------------------------------------------

def check_token_shadowing(
    tokenizer: _Tokenizer,
    candidate_tokens: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Verify that newly added vocabulary entries are recognised as single
    tokens and not silently fragmented (shadowed) by the underlying
    WordPiece algorithm.

    Args:
        tokenizer:        The augmented Track A tokenizer.
        candidate_tokens: Tokens that were added (e.g. tok_a.http_tokens).

    Returns:
        Dict keyed by candidate token::

          {
            "UNION SELECT": {
              "n_fragments": 1,
              "fragments": ["UNION SELECT"],
              "shadowed": False,
            },
            "WAITFOR DELAY": {
              "n_fragments": 3,
              "fragments": ["WAIT", "##FOR", "DE", "##LAY"],
              "shadowed": True,
            },
          }

    A token is considered *shadowed* when n_fragments > 1, meaning the
    tokenizer's WordPiece scoring still prefers to split it rather than
    treating it as an atomic unit.
    """
    results: dict[str, dict[str, Any]] = {}
    for token in candidate_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            fragments = tokenizer.convert_ids_to_tokens(ids)
        except Exception as exc:
            results[token] = {"error": str(exc), "shadowed": None}
            continue

        results[token] = {
            "n_fragments": len(fragments),
            "fragments":   fragments,
            "shadowed":    len(fragments) > 1,
        }
    return results


def summarise_shadowing(shadow_report: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate statistics over a shadowing report."""
    total   = len(shadow_report)
    shadowed = [t for t, v in shadow_report.items() if v.get("shadowed") is True]
    return {
        "total_checked":   total,
        "n_shadowed":      len(shadowed),
        "shadow_rate":     round(len(shadowed) / max(total, 1), 4),
        "shadowed_tokens": shadowed,
    }


# ---------------------------------------------------------------------------
# 6. Side-by-side comparison report
# ---------------------------------------------------------------------------

def build_comparison_report(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a structured side-by-side comparison between Track A and B.
    Determines winners per metric and computes deltas.
    """
    def _winner(key: str, lower_is_better: bool = True) -> str:
        va = metrics_a.get(key, float("inf"))
        vb = metrics_b.get(key, float("inf"))
        if va == vb:
            return "tie"
        if lower_is_better:
            return "track_a" if va < vb else "track_b"
        return "track_a" if va > vb else "track_b"

    def _delta(key: str) -> float | None:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None or vb is None:
            return None
        return round(float(vb) - float(va), 6)   # positive = track_b is higher

    metrics_to_compare = [
        ("oov_rate",           True),   # lower is better
        ("fertility",          True),
        ("truncation_rate",    True),
        ("avg_seq_len",        True),
        ("subword_char_ratio", False),  # higher is better (more compression)
    ]

    winners = {k: _winner(k, lib) for k, lib in metrics_to_compare}
    deltas  = {f"delta_{k}": _delta(k) for k, _ in metrics_to_compare}

    # Vocab Jaccard — only if both expose a `vocab` attribute
    jaccard: float | None = None
    try:
        vocab_a = set(metrics_a.get("_vocab_a", []))
        vocab_b = set(metrics_b.get("_vocab_b", []))
        if vocab_a and vocab_b:
            jaccard = round(len(vocab_a & vocab_b) / len(vocab_a | vocab_b), 4)
    except Exception:
        pass

    return {
        "track_a": {k: metrics_a.get(k) for k, _ in metrics_to_compare},
        "track_b": {k: metrics_b.get(k) for k, _ in metrics_to_compare},
        "winners": winners,
        "deltas":  deltas,
        "vocab_jaccard_overlap": jaccard,
        "recommendation": _recommend(winners),
    }


def _recommend(winners: dict[str, str]) -> str:
    """Heuristic recommendation based on winner tallies."""
    tally: dict[str, int] = defaultdict(int)
    for track in winners.values():
        tally[track] += 1
    if not tally:
        return "inconclusive"
    best = max(tally, key=tally.__getitem__)
    return best if tally[best] > len(winners) // 2 else "inconclusive"