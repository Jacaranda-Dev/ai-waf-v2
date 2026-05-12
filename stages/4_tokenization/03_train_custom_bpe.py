"""
stages/4_tokenization/03_train_custom_bpe.py
--------------------------------------------
Stage 3.3 — Train Track B custom BPE tokenizer on HTTP corpus,
then evaluate with the unified metric schema.

Enhancements over original:
  * Delimiter-preserving normalization: CRLF / LF / CR sequences are
    replaced with dedicated special tokens ([CRLF], [LF], [CR]) before
    corpus writing. This preserves the structural boundaries between
    HTTP headers and body, allowing BPE to learn transitions that
    correlate with attack patterns.
  * Stratified corpus sampling: training corpus uses a reproducible
    random draw rather than an arbitrary head-slice.
  * Evaluation uses the shared tokenizer_eval module (no logic drift).
  * Comparison with Track A deferred to script 05 to avoid embedding
    partial comparison logic here.

Run:
    python stages/4_tokenization/03_train_custom_bpe.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

from tokenizer_eval import compute_full_metrics, stratified_sample

log = get_logger(__name__)

# Special tokens used in lieu of raw control characters.
# These must also be registered in HttpTokenizer.SPECIAL_TOKENS so the
# BPE model treats them as indivisible units.
CRLF_TOKEN = "[CRLF]"
LF_TOKEN   = "[LF]"
CR_TOKEN   = "[CR]"

CORPUS_SAMPLE_SIZE = 500_000
EVAL_SAMPLE_SIZE   = 5_000


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def _normalize_delimiters(text: str) -> str:
    """
    Replace HTTP protocol delimiters with dedicated special tokens.

    CRLF (\\r\\n) must be checked BEFORE the individual \\r and \\n
    replacements to avoid double-substitution.

    This preserves the structural boundary information that raw whitespace
    replacement destroys — critical for the BPE algorithm to learn
    header / body transitions.
    """
    text = text.replace("\r\n", f" {CRLF_TOKEN} ")  # HTTP spec CRLF — check first
    text = text.replace("\n",   f" {LF_TOKEN} ")     # Unix newlines
    text = text.replace("\r",   f" {CR_TOKEN} ")     # Stray carriage returns
    return text


def build_corpus(
    splits_dir: Path,
    corpus_path: Path,
    sample_n: int = CORPUS_SAMPLE_SIZE,
    seed: int = 42,
) -> None:
    """
    Build a plain-text HTTP corpus for BPE training.

    One normalised request per line; CRLF boundaries are preserved as
    special tokens rather than collapsed into whitespace.
    """
    log.info("Building tokenizer training corpus...")
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    texts = pq.read_table(train_path, columns=["raw"])["raw"].to_pylist()

    if len(texts) > sample_n:
        rng = random.Random(seed)
        rng.shuffle(texts)
        texts = texts[:sample_n]

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8", errors="replace") as fh:
        for text in texts:
            fh.write(_normalize_delimiters(text) + "\n")

    log.info(f"Corpus written: {len(texts):,} samples → {corpus_path}")
    log.info(
        f"Delimiter tokens used: '{CRLF_TOKEN}', '{LF_TOKEN}', '{CR_TOKEN}' "
        "(structural boundaries preserved)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    tok_cfg     = cfg.tokenizer.track_b
    splits_dir  = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = Path(tok_cfg.output_dir) / "train_corpus.txt"

    # ------------------------------------------------------------------
    # Build corpus with delimiter preservation
    # ------------------------------------------------------------------
    build_corpus(splits_dir, corpus_path, seed=cfg.project.seed)

    # ------------------------------------------------------------------
    # Train BPE tokenizer
    # ------------------------------------------------------------------
    log.info(f"Training BPE tokenizer (vocab_size={tok_cfg.vocab_size})...")
    tokenizer_b = HttpTokenizer.train(
        corpus_path=corpus_path,
        vocab_size=tok_cfg.vocab_size,
        output_dir=tok_cfg.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )
    log.info(f"Track B tokenizer trained — vocab_size={tokenizer_b.vocab_size}")

    # ------------------------------------------------------------------
    # Stratified evaluation on val split
    # ------------------------------------------------------------------
    val_table  = pq.read_table(splits_dir / "val.parquet", columns=["raw", "attack_class"])
    all_texts  = val_table["raw"].to_pylist()
    all_labels = val_table["attack_class"].to_pylist()

    texts, labels = stratified_sample(
        all_texts, all_labels,
        n=min(EVAL_SAMPLE_SIZE, len(all_texts)),
        seed=cfg.project.seed,
    )
    log.info(
        f"Stratified sample: {len(texts)} requests across "
        f"{len(set(labels))} attack classes"
    )

    # ------------------------------------------------------------------
    # Full unified metrics
    # ------------------------------------------------------------------
    seq_len  = cfg.tokenizer.seq_len
    result_b = compute_full_metrics(
        tokenizer_b, texts, labels,
        seq_len=seq_len,
        unk_token="[UNK]",
        track_name="track_b",
    )

    log.info(
        f"Track B | oov={result_b['oov_rate']:.4f}  "
        f"fertility={result_b['fertility']:.4f}  "
        f"truncation={result_b['truncation_rate']:.4f}  "
        f"subword_char_ratio={result_b['subword_char_ratio']:.2f}"
    )
    for cls, m in result_b["per_class"].items():
        log.info(
            f"  {cls:<22s} oov={m['oov_rate']:.4f}  "
            f"fertility={m['fertility']:.4f}  "
            f"trunc={m['truncation_rate']:.4f}  n={m['n_samples']}"
        )

    # ------------------------------------------------------------------
    # Persist — full comparison deferred to 05_compare_tokenizers.py
    # ------------------------------------------------------------------
    out_path = reports_dir / "tokenizer_stats_track_b.json"
    out_path.write_text(json.dumps(result_b, indent=2))
    log.info(f"Track B stats saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Track B BPE tokenizer.")
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())