"""
stages/3_data_augmentation/11_format_validation.py  (merged with 12, 13, 14)
------------------------------------------------------------------
Stage 2.6 — Quality filtering pipeline.

Runs four passes over all augmented data:
  1. Format validation  : must parse as valid HTTP structure
  2. Tokenizer coverage : reject if >max_unk_ratio of tokens are [UNK]
  3. Semantic dedup     : remove near-duplicates at 0.90 Jaccard
  4. Label consistency  : flag samples where CRS-based heuristics
                          conflict with the assigned label

Input:  all Parquet files in data/augmented/
Output: data/filtered/filtered.parquet  (passes all checks)
        data/filtered/rejected.parquet  (fails any check)

Run:
    python stages/3_data_augmentation/11_format_validation.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from ai_waf_v2.data.schema import HttpRecord, PARQUET_SCHEMA
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# Simple CRS-like heuristics for label consistency check
# These are deliberately conservative to avoid over-flagging
_SQLI_PATTERNS = re.compile(
    r"(UNION\s+SELECT|SELECT\s+.*FROM|DROP\s+TABLE|INSERT\s+INTO|"
    r"OR\s+1=1|AND\s+1=1|SLEEP\s*\(|BENCHMARK\s*\(|WAITFOR\s+DELAY)",
    re.IGNORECASE,
)
_XSS_PATTERNS = re.compile(
    r"(<script|onerror\s*=|onload\s*=|javascript:|<iframe|alert\s*\()",
    re.IGNORECASE,
)
_LFI_PATTERNS = re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|%252e)", re.IGNORECASE)


def _has_attack_heuristic(raw: str) -> bool:
    return bool(
        _SQLI_PATTERNS.search(raw)
        or _XSS_PATTERNS.search(raw)
        or _LFI_PATTERNS.search(raw)
    )


def _is_valid_http(record: HttpRecord) -> tuple[bool, str]:
    """Basic HTTP structure validation."""
    if not record.raw:
        return False, "empty_raw"
    if len(record.raw) < 10:
        return False, "too_short"
    if not re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s", record.raw):
        return False, "invalid_method"
    if len(record.raw) > 8192:
        return False, "too_long"
    return True, ""


def _make_minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(max(1, len(text) - 2)):
        m.update(text[i:i+3].encode("utf-8", errors="replace"))
    return m


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    filt_cfg = cfg.augmentation.filtering

    # Load tokenizer for UNK rate check
    try:
        tokenizer = HttpTokenizer.load(
            cfg.tokenizer.track_b.output_dir,
            seq_len=cfg.tokenizer.seq_len,
        )
        has_tokenizer = True
    except FileNotFoundError:
        log.warning("Track B tokenizer not found — skipping UNK rate filter")
        has_tokenizer = False

    # Collect all augmented Parquet files
    aug_dir   = Path(cfg.paths.data_augmented)
    parquet_files = list(aug_dir.rglob("*.parquet"))

    # Also include the base deduped normalized data
    base_path = Path(cfg.paths.data_normalized) / "deduped.parquet"
    if base_path.exists():
        parquet_files.append(base_path)

    if not parquet_files:
        log.error("No Parquet files found in data/augmented/ — run augmentation stages first")
        return

    log.info(f"Loading {len(parquet_files)} Parquet files...")

    dfs = []
    for fp in parquet_files:
        try:
            dfs.append(pq.read_table(fp).to_pandas())
        except Exception as e:
            log.warning(f"Failed to load {fp}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Total records before filtering: {len(df):,}")

    records        = [HttpRecord.from_dict(row) for _, row in df.iterrows()]
    passed:  list[HttpRecord] = []
    rejected: list[tuple[HttpRecord, str]] = []

    # ── Pass 1: format validation ─────────────────
    log.info("Pass 1: HTTP format validation...")
    after_fmt: list[HttpRecord] = []
    for r in records:
        ok, reason = _is_valid_http(r)
        if ok:
            after_fmt.append(r)
        else:
            rejected.append((r, f"format:{reason}"))

    log.info(f"  Passed: {len(after_fmt):,}  Rejected: {len(records)-len(after_fmt):,}")

    # ── Pass 2: tokenizer UNK rate ────────────────
    log.info("Pass 2: tokenizer coverage check...")
    after_tok: list[HttpRecord] = []
    if has_tokenizer:
        unk_id = tokenizer.token_to_id("[UNK]") or 1
        pad_id = tokenizer.pad_token_id
        max_unk = filt_cfg.max_unk_ratio

        for r in after_fmt:
            enc   = tokenizer.encode(r.raw)
            ids   = [i for i in enc.ids if i != pad_id]
            unk_r = sum(1 for i in ids if i == unk_id) / max(1, len(ids))
            if unk_r <= max_unk:
                after_tok.append(r)
            else:
                rejected.append((r, f"high_unk_rate:{unk_r:.2f}"))
    else:
        after_tok = after_fmt

    log.info(f"  Passed: {len(after_tok):,}  Rejected: {len(after_fmt)-len(after_tok):,}")

    # ── Pass 3: semantic deduplication ────────────
    log.info("Pass 3: semantic deduplication (MinHash LSH)...")
    threshold = cfg.data.dedup.semantic_threshold
    lsh       = MinHashLSH(threshold=threshold, num_perm=64)
    after_dedup: list[HttpRecord] = []

    for r in after_tok:
        key = r.id
        mh  = _make_minhash(r.raw)
        if not lsh.query(mh):
            lsh.insert(key, mh)
            after_dedup.append(r)
        else:
            rejected.append((r, "semantic_duplicate"))

    log.info(f"  Passed: {len(after_dedup):,}  Rejected: {len(after_tok)-len(after_dedup):,}")

    # ── Pass 4: label consistency ─────────────────
    log.info("Pass 4: label consistency check...")
    n_conflicts = 0
    if filt_cfg.label_consistency_check:
        for r in after_dedup:
            heuristic_malicious = _has_attack_heuristic(r.raw)
            # Flag when heuristic strongly disagrees with label
            # Only reject clear contradictions: label=0 but strong attack signal
            if r.label == 0 and heuristic_malicious:
                # Don't reject — just flag. Benign edge-cases legitimately
                # contain SQL keywords. We keep them but note the conflict.
                n_conflicts += 1
                # Override: keep in passed set with a conflict note in source
                object.__setattr__(r, "source", r.source + "_conflict_flagged")

        log.info(f"  Label conflicts flagged (kept): {n_conflicts:,}")

    passed = after_dedup

    # ── Write outputs ─────────────────────────────
    filtered_dir = Path(cfg.paths.data_filtered)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    passed_df  = pd.DataFrame([r.model_dump() for r in passed])
    pq.write_table(
        pa.Table.from_pandas(passed_df, schema=PARQUET_SCHEMA, preserve_index=False),
        filtered_dir / "filtered.parquet",
        compression="snappy",
    )

    rejected_df = pd.DataFrame([r.model_dump() for r, _ in rejected])
    if not rejected_df.empty:
        pq.write_table(
            pa.Table.from_pandas(rejected_df, schema=PARQUET_SCHEMA, preserve_index=False),
            filtered_dir / "rejected.parquet",
            compression="snappy",
        )

    n_passed   = len(passed)
    n_rejected = len(rejected)
    log.info(
        f"Filtering complete: {n_passed:,} passed, {n_rejected:,} rejected "
        f"({n_rejected/(n_passed+n_rejected)*100:.1f}% rejection rate)"
    )

    stats = {
        "n_input":    len(records),
        "n_passed":   n_passed,
        "n_rejected": n_rejected,
        "rejection_rate": round(n_rejected / max(1, len(records)), 4),
        "n_label_conflicts": n_conflicts,
        "rejection_reasons": _count_reasons(rejected),
    }
    stats_path = Path(cfg.paths.reports) / "metrics" / "filtering_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))


def _count_reasons(rejected: list[tuple]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, reason in rejected:
        top = reason.split(":")[0]
        counts[top] = counts.get(top, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())