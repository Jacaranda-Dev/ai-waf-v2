"""
stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py
----------------------------------------------------------------
 Cross-dataset deduplication of the canonical corpus.

Two-pass strategy:
  Pass 1 — Exact SHA-256 hash on the raw HTTP string.
            Eliminates identical duplicates in O(n) time.
  Pass 2 — MinHash LSH at a configurable Jaccard threshold (default 0.85).
            Eliminates near-duplicates (e.g. CSIC 2010 vs SR-BH payloads
            that differ only in whitespace or minor header variation).

Why this is a standalone stage
───────────────────────────────
MinHash LSH over millions of records is the most memory-intensive step in
the entire curation pipeline.  Isolating it here lets the OS reclaim its
working set before the next stage starts, preventing OOM errors from
cascading into reporting or splitting.  Do NOT merge this into the
acquire/normalise stage or the corpus report.

Inputs:   data/normalized/all_datasets.parquet   (written by Stage 01)
Outputs:  data/normalized/deduped.parquet
          reports/metrics/dedup_stats.json

Run:
    python stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py \\
        --config config/pipeline.yaml

    # override Jaccard threshold at the CLI without touching config:
    python ... --threshold 0.90

    # skip MinHash (exact dedup only) — useful for quick smoke-tests:
    python ... --exact-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
from datasketch import MinHash, MinHashLSH

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# Number of permutations for MinHash — higher → more accurate, more RAM.
# 128 gives ~3% estimation error at the configured threshold.
_NUM_PERM = 128


# ─────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _make_minhash(text: str) -> MinHash:
    """Build a MinHash from character 3-grams of `text`."""
    mh = MinHash(num_perm=_NUM_PERM)
    for i in range(max(1, len(text) - 2)):
        mh.update(text[i : i + 3].encode("utf-8"))
    return mh


# ─────────────────────────────────────────────────────────
# Deduplication passes
# ─────────────────────────────────────────────────────────

def _exact_dedup(table: pa.Table) -> tuple[pa.Table, int]:
    """
    Pass 1: drop rows whose SHA-256(raw) has already been seen.

    Operates on the PyArrow table directly (no full pandas copy) to keep
    peak RAM as low as possible before the MinHash pass.

    Returns (deduplicated table, n_removed).
    """
    log.info("Pass 1: exact SHA-256 deduplication…")
    raws   = table.column("raw").to_pylist()
    seen:  set[str] = set()
    keep:  list[bool] = []

    for raw in raws:
        h = _sha256(str(raw))
        if h in seen:
            keep.append(False)
        else:
            seen.add(h)
            keep.append(True)

    import pyarrow.compute as pc
    mask          = pa.array(keep, type=pa.bool_())
    deduped       = table.filter(mask)
    n_removed     = len(table) - len(deduped)
    log.info(
        f"  After exact dedup: {len(deduped):,} records "
        f"({n_removed:,} removed)"
    )
    return deduped, n_removed


def _minhash_dedup(table: pa.Table, threshold: float) -> tuple[pa.Table, int]:
    """
    Pass 2: drop near-duplicates using MinHash LSH.

    Iterates once over the table; for each row that has no near-neighbour
    already in the LSH index, insert it and mark it for keeping.

    Returns (deduplicated table, n_removed).
    """
    log.info(f"Pass 2: MinHash LSH deduplication (threshold={threshold})…")
    raws  = table.column("raw").to_pylist()
    lsh   = MinHashLSH(threshold=threshold, num_perm=_NUM_PERM)
    keep: list[bool] = []

    for i, raw in enumerate(raws):
        mh = _make_minhash(str(raw))
        if lsh.query(mh):
            keep.append(False)
        else:
            lsh.insert(str(i), mh)
            keep.append(True)

    import pyarrow.compute as pc
    mask      = pa.array(keep, type=pa.bool_())
    deduped   = table.filter(mask)
    n_removed = len(table) - len(deduped)
    log.info(
        f"  After MinHash LSH dedup: {len(deduped):,} records "
        f"({n_removed:,} near-duplicates removed)"
    )
    return deduped, n_removed


# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    normalized_dir = Path(cfg.paths.data_normalized)
    in_path        = normalized_dir / "all_datasets.parquet"
    out_path       = normalized_dir / "deduped.parquet"

    if not in_path.exists():
        log.error(f"Input not found: {in_path} — run 01_acquire_and_normalize.py first")
        return

    table  = pq.read_table(in_path)
    n_orig = len(table)
    log.info(f"Loaded {n_orig:,} records for deduplication")

    # ── Pass 1: exact hash ───────────────────────────────────────────────────
    table, removed_exact = _exact_dedup(table)
    n_after_exact        = len(table)

    # ── Pass 2: MinHash LSH (optional) ──────────────────────────────────────
    removed_minhash = 0
    if not args.exact_only:
        threshold = args.threshold or cfg.data.dedup.minhash_threshold
        table, removed_minhash = _minhash_dedup(table, threshold)
    else:
        log.info("Pass 2: skipped (--exact-only)")
        threshold = None

    n_final = len(table)

    # ── Write output ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    log.info(f"Wrote {n_final:,} records → {out_path}")

    # ── Persist stats ─────────────────────────────────────────────────────────
    stats = {
        "n_original":          n_orig,
        "n_after_exact_dedup": n_after_exact,
        "n_after_minhash":     n_final,
        "removed_exact":       removed_exact,
        "removed_minhash":     removed_minhash,
        "total_removed":       n_orig - n_final,
        "dedup_rate":          round((n_orig - n_final) / max(1, n_orig), 4),
        "minhash_threshold":   threshold,
        "exact_only":          args.exact_only,
    }
    stats_path = Path(cfg.paths.reports) / "metrics" / "dedup_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(
        f"Dedup complete — {n_orig - n_final:,} records removed "
        f"({stats['dedup_rate']:.1%}). Stats → {stats_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 — Cross-dataset deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",     default="config/pipeline.yaml")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="MinHash Jaccard threshold (overrides config; default: cfg.data.dedup.minhash_threshold)",
    )
    p.add_argument(
        "--exact-only",
        action="store_true",
        help="Run only SHA-256 exact dedup; skip the MinHash LSH pass",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())