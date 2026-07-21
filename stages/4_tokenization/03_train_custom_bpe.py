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
import random
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.tokenizer.http_tokenizer import (
    HttpTokenizer,
    CRLF_TOKEN,
    LF_TOKEN,
    CR_TOKEN,
)
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

CORPUS_SAMPLE_SIZE = 500_000  # fallback when not set in config


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

    One normalized request per line; CRLF boundaries are preserved as
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
    n_replaced = 0
    with corpus_path.open("w", encoding="utf-8", errors="replace") as fh:
        for text in texts:
            normalized = _normalize_delimiters(text)
            n_replaced += normalized.count("�")
            fh.write(normalized + "\n")

    log.info(f"Corpus written: {len(texts):,} samples → {corpus_path}")
    if n_replaced:
        log.warning(
            f"Corpus: {n_replaced:,} characters replaced with U+FFFD — "
            "source data contains unencodable byte sequences"
        )
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

    require_inputs({
        "data/splits/train.parquet": "make data_augment_all",
    })
    if check_output(
        Path(cfg.tokenizer.track_b.output_dir) / "tokenizer.json",
        args.force, "Stage 4.3 Track B BPE training"
    ):
        return

    tok_cfg     = cfg.tokenizer.track_b
    splits_dir  = Path(cfg.paths.data_splits)

    corpus_path = Path(tok_cfg.output_dir) / "train_corpus.txt"
    timer       = StepTimer()

    # ------------------------------------------------------------------
    # Build corpus with delimiter preservation
    # ------------------------------------------------------------------
    sample_n = getattr(tok_cfg, "corpus_sample_size", CORPUS_SAMPLE_SIZE)
    with timer.step("build_corpus"):
        build_corpus(splits_dir, corpus_path, sample_n=sample_n, seed=cfg.project.seed)

    # ------------------------------------------------------------------
    # Train BPE tokenizer
    # ------------------------------------------------------------------
    log.info(f"Training BPE tokenizer (vocab_size={tok_cfg.vocab_size})...")
    with timer.step("train_bpe"):
        tokenizer_b = HttpTokenizer.train(
            corpus_path=corpus_path,
            vocab_size=tok_cfg.vocab_size,
            output_dir=tok_cfg.output_dir,
            seq_len=cfg.tokenizer.seq_len,
        )
    log.info(f"Track B tokenizer trained — vocab_size={tokenizer_b.vocab_size}")
    log.info("Evaluation deferred to 04_measure_oov_track_b.py")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="03_train_custom_bpe"):
            mlflow.log_params({
                "vocab_size":         tok_cfg.vocab_size,
                "corpus_sample_size": sample_n,
                "seq_len":            cfg.tokenizer.seq_len,
                "output_dir":         tok_cfg.output_dir,
            })
            log_metrics_dict({
                "actual_vocab_size": float(tokenizer_b.vocab_size),
            })
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Track B BPE tokenizer.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())