"""
stages/4_tokenization/03_train_custom_bpe.py
-----------------------------------------
Stage 3.2-3.4 — Train Track B custom BPE tokenizer on HTTP corpus,
then evaluate and compare with Track A.

Run:
    python stages/4_tokenization/03_train_custom_bpe.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.tokenizer.vocab_utils import compare_tokenizers, measure_oov
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


def build_corpus(splits_dir: Path, corpus_path: Path, sample_n: int = 500_000) -> None:
    """
    Build a plain-text HTTP corpus file for SentencePiece training.
    One request per line from the train split.
    """
    log.info("Building tokenizer training corpus...")
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    table = pq.read_table(train_path, columns=["raw"])
    texts = table["raw"].to_pylist()

    if len(texts) > sample_n:
        import random
        random.shuffle(texts)
        texts = texts[:sample_n]

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8", errors="replace") as f:
        for text in texts:
            # One request per line, strip embedded newlines within the request
            f.write(text.replace("\n", " ").replace("\r", " ") + "\n")

    log.info(f"Corpus written: {len(texts):,} samples → {corpus_path}")


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    tok_cfg    = cfg.tokenizer.track_b
    splits_dir = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = Path(tok_cfg.output_dir) / "train_corpus.txt"

    # Build corpus from train split
    build_corpus(splits_dir, corpus_path)

    # Train BPE tokenizer
    log.info(f"Training BPE tokenizer (vocab_size={tok_cfg.vocab_size})...")
    tokenizer_b = HttpTokenizer.train(
        corpus_path=corpus_path,
        vocab_size=tok_cfg.vocab_size,
        output_dir=tok_cfg.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )
    log.info(f"Track B tokenizer trained. Vocab size: {tokenizer_b.vocab_size}")

    # ── Evaluation on val split ───────────────────
    val_table  = pq.read_table(splits_dir / "val.parquet", columns=["raw", "attack_class"])
    val_texts  = val_table["raw"].to_pylist()[:5000]   # sample for speed
    val_labels = val_table["attack_class"].to_pylist()[:5000]

    log.info("Measuring Track B OOV and fertility...")
    stats_b = measure_oov(tokenizer_b, val_texts)
    log.info(
        f"Track B: oov_rate={stats_b['oov_rate']:.4f}  "
        f"avg_seq_len={stats_b['avg_seq_len']:.1f}  "
        f"fertility={stats_b['fertility']:.4f}"
    )

    # Per-attack-class OOV breakdown
    per_class_oov: dict[str, float] = {}
    from collections import defaultdict
    class_texts: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(val_texts, val_labels):
        class_texts[label].append(text)

    for cls, texts in class_texts.items():
        oov = measure_oov(tokenizer_b, texts)["oov_rate"]
        per_class_oov[cls] = round(oov, 6)
        log.info(f"  {cls:20s}: OOV={oov:.4f}")

    # ── Track A comparison (if tokenizer exists) ──
    comparison: dict = {}
    track_a_dir = Path(cfg.tokenizer.track_a.output_dir)
    if (track_a_dir / "tokenizer_config.json").exists():
        log.info("Comparing with Track A tokenizer...")
        from transformers import AutoTokenizer
        tokenizer_a = AutoTokenizer.from_pretrained(str(track_a_dir))
        comparison  = compare_tokenizers(
            tokenizer_a, tokenizer_b,
            val_texts, val_labels,
        )
        log.info(
            f"Track A: oov_rate={comparison['track_a']['oov_rate']:.4f}  "
            f"fertility={comparison['track_a']['fertility']:.4f}"
        )
        log.info(
            f"Winner OOV: {comparison['winner_oov']}  "
            f"Winner fertility: {comparison['winner_fertility']}"
        )
    else:
        log.info("Track A tokenizer not found — skipping comparison (run 01_augment_pretrained_vocab.py first)")

    # Save stats
    result = {
        "track_b": {**stats_b, "per_class_oov": per_class_oov},
        "comparison": comparison,
        "vocab_size": tokenizer_b.vocab_size,
        "seq_len":    cfg.tokenizer.seq_len,
    }
    (reports_dir / "tokenizer_stats.json").write_text(
        json.dumps(result, indent=2)
    )
    log.info(f"Tokenizer stats saved to {reports_dir / 'tokenizer_stats.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())