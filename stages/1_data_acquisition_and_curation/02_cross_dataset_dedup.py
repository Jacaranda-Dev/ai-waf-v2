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
acquire/Normalize stage or the corpus report.

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

    # force re-dedup even if deduped.parquet already exists:
    python ... --force-dedup
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import pyarrow as pa
from datasketch import MinHash, MinHashLSH

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

# Number of permutations for MinHash — higher → more accurate, more RAM.
# 128 gives ~3% estimation error at the configured threshold.
_NUM_PERM = 128

# Minimum unique character 3-grams required before a payload is eligible for
# MinHash comparison.  Below this count the banding estimate has high variance
# and produces false positives (e.g. two different short paths sharing a long
# common prefix).  Exact dedup (Pass 1) handles true duplicates at any length.
_MIN_SHINGLES = 40


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

def _exact_dedup(table: pa.Table, n_samples: int = 3) -> tuple[pa.Table, int, list[dict]]:
    """
    Pass 1: drop rows whose SHA-256(raw) has already been seen.

    Operates on the PyArrow table directly (no full pandas copy) to keep
    peak RAM as low as possible before the MinHash pass.

    Returns (deduplicated table, n_removed, sample_pairs).
    Each sample_pair is {"original": {...}, "duplicate": {...}}.
    """
    log.info("Pass 1: exact SHA-256 deduplication…")
    raws    = table.column("raw").to_pylist()
    sources = table.column("source").to_pylist()
    labels  = table.column("label").to_pylist()
    classes = table.column("attack_class").to_pylist()

    seen: dict[str, int] = {}  # hash → first-seen row index
    keep: list[bool] = []
    samples: list[dict] = []

    for i, raw in enumerate(raws):
        h = _sha256(str(raw))
        if h in seen:
            keep.append(False)
            if len(samples) < n_samples:
                orig_i = seen[h]
                samples.append({
                    "original":  {"idx": orig_i, "source": sources[orig_i],
                                  "label": labels[orig_i], "attack_class": classes[orig_i],
                                  "raw": str(raws[orig_i])},
                    "duplicate": {"idx": i,      "source": sources[i],
                                  "label": labels[i],      "attack_class": classes[i],
                                  "raw": str(raw)},
                })
        else:
            seen[h] = i
            keep.append(True)

    import pyarrow.compute as pc
    mask      = pa.array(keep, type=pa.bool_())
    deduped   = table.filter(mask)
    n_removed = len(table) - len(deduped)
    log.info(
        f"  After exact dedup: {len(deduped):,} records "
        f"({n_removed:,} removed)"
    )
    return deduped, n_removed, samples


def _minhash_dedup(table: pa.Table, threshold: float, n_samples: int = 3) -> tuple[pa.Table, int, list[dict]]:
    """
    Pass 2: drop near-duplicates using MinHash LSH.

    Iterates once over the table; for each row that has no near-neighbour
    already in the LSH index, insert it and mark it for keeping.

    Returns (deduplicated table, n_removed, sample_pairs).
    Each sample_pair is {"kept": {...}, "removed": {...}}.
    """
    log.info(f"Pass 2: MinHash LSH deduplication (threshold={threshold})…")
    raws    = table.column("raw").to_pylist()
    methods = table.column("method").to_pylist()
    paths   = table.column("path").to_pylist()
    queries = table.column("query_string").to_pylist()
    bodies  = table.column("body").to_pylist()
    sources = table.column("source").to_pylist()
    labels  = table.column("label").to_pylist()
    classes = table.column("attack_class").to_pylist()

    lsh             = MinHashLSH(threshold=threshold, num_perm=_NUM_PERM)
    minhashes:      dict[str, MinHash] = {}  # key → minhash, for sample Jaccard
    methods_by_key: dict[str, str]    = {}   # key → HTTP method of inserted record
    keep:           list[bool]         = []
    samples:        list[dict]         = []

    for i, raw in enumerate(raws):
        # Fingerprint on path+query+body; headers omitted because shared
        # User-Agent/Accept strings inflate Jaccard across unrelated requests.
        # Strip scheme+host from path: CSIC 2010 stores absolute URLs
        # (http://localhost:8080/path) so the 21-char shared prefix inflates
        # Jaccard for all CSIC pairs — even unrelated endpoints.
        path_norm = urlparse(paths[i]).path or paths[i]
        payload  = f"{path_norm}?{queries[i]}\n{bodies[i]}"
        shingles = {payload[j : j + 3] for j in range(max(1, len(payload) - 2))}

        if len(shingles) < _MIN_SHINGLES:
            # Too few unique 3-grams for a reliable MinHash estimate — always
            # keep and skip LSH insertion (prevents short-path false positives
            # where a long shared prefix dominates the Jaccard).
            keep.append(True)
            continue

        mh      = _make_minhash(payload)
        matches = lsh.query(mh)

        # Restrict to same HTTP method.  GET/POST pairs for the same CSIC 2010
        # form endpoint share all params (query vs body), giving Jaccard > 0.85
        # even though they are distinct HTTP interactions.
        same_method = [m for m in matches if methods_by_key[m] == methods[i]]

        if same_method:
            keep.append(False)
            if len(samples) < n_samples:
                match_key = same_method[0]
                match_i   = int(match_key)
                jaccard   = mh.jaccard(minhashes[match_key])
                samples.append({
                    "kept": {
                        "idx": match_i, "source": sources[match_i],
                        "label": labels[match_i], "attack_class": classes[match_i],
                        "raw": str(raws[match_i]),
                    },
                    "removed": {
                        "idx": i, "source": sources[i],
                        "label": labels[i], "attack_class": classes[i],
                        "raw": str(raw),
                        "jaccard": round(jaccard, 4),
                    },
                })
        else:
            key                = str(i)
            lsh.insert(key, mh)
            minhashes[key]      = mh
            methods_by_key[key] = methods[i]
            keep.append(True)

    import pyarrow.compute as pc
    mask      = pa.array(keep, type=pa.bool_())
    deduped   = table.filter(mask)
    n_removed = len(table) - len(deduped)
    log.info(
        f"  After MinHash LSH dedup: {len(deduped):,} records "
        f"({n_removed:,} near-duplicates removed)"
    )
    return deduped, n_removed, samples


# ─────────────────────────────────────────────────────────
# Sample writer
# ─────────────────────────────────────────────────────────

_RAW_PREVIEW = 200  # characters to show per raw field


def _write_dedup_samples(
    exact_samples: list[dict],
    minhash_samples: list[dict],
    out_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("PASS 1 — EXACT SHA-256 DUPLICATES")
    lines.append("=" * 72)
    if not exact_samples:
        lines.append("(none found)")
    for n, pair in enumerate(exact_samples, 1):
        orig = pair["original"]
        dup  = pair["duplicate"]
        lines += [
            f"\n[{n}] ORIGINAL  idx={orig['idx']}  source={orig['source']!r}"
            f"  label={orig['label']}  class={orig['attack_class']!r}",
            f"    {orig['raw'][:_RAW_PREVIEW]}",
            f"    DUPLICATE  idx={dup['idx']}  source={dup['source']!r}",
            f"    {dup['raw'][:_RAW_PREVIEW]}",
        ]

    lines.append("\n" + "=" * 72)
    lines.append("PASS 2 — MINHASH LSH NEAR-DUPLICATES")
    lines.append("=" * 72)
    if not minhash_samples:
        lines.append("(none found)")
    for n, pair in enumerate(minhash_samples, 1):
        kept    = pair["kept"]
        removed = pair["removed"]
        lines += [
            f"\n[{n}] KEPT     idx={kept['idx']}  source={kept['source']!r}"
            f"  label={kept['label']}  class={kept['attack_class']!r}",
            f"    {kept['raw'][:_RAW_PREVIEW]}",
            f"    REMOVED  idx={removed['idx']}  source={removed['source']!r}"
            f"  jaccard={removed['jaccard']}",
            f"    {removed['raw'][:_RAW_PREVIEW]}",
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Dedup samples → {out_path}")


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

    if out_path.exists() and not args.force_dedup:
        try:
            n_rows = pq.read_metadata(out_path).num_rows
        except Exception:
            n_rows = 0
        if n_rows > 0:
            log.info(
                f"Deduped file exists ({n_rows:,} rows) — skipping. "
                "Use --force-dedup to re-run."
            )
            return

    table  = pq.read_table(in_path)
    n_orig = len(table)
    log.info(f"Loaded {n_orig:,} records for deduplication")
    timer  = StepTimer()

    # ── Pass 1: exact hash ───────────────────────────────────────────────────
    with timer.step("exact_dedup"):
        table, removed_exact, exact_samples = _exact_dedup(table)
    n_after_exact = len(table)

    # ── Pass 2: MinHash LSH (optional) ──────────────────────────────────────
    removed_minhash  = 0
    minhash_samples: list[dict] = []
    if not args.exact_only:
        threshold = args.threshold or cfg.data.dedup.minhash_threshold
        with timer.step("minhash_dedup"):
            table, removed_minhash, minhash_samples = _minhash_dedup(table, threshold)
    else:
        log.info("Pass 2: skipped (--exact-only)")
        threshold = None

    # ── Write sample duplicates ──────────────────────────────────────────────
    samples_path = Path(cfg.paths.data_normalized) / "dedup_samples.txt"
    _write_dedup_samples(exact_samples, minhash_samples, samples_path)

    n_final = len(table)

    # ── Write output ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with timer.step("parquet_write"):
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
        "timings_s":           timer.timings,
    }
    stats_path = Path(cfg.paths.reports) / "metrics" / "dedup_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(
        f"Dedup complete — {n_orig - n_final:,} records removed "
        f"({stats['dedup_rate']:.1%}). Stats → {stats_path}"
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        import mlflow
        import pyarrow.compute as pc

        init_experiment(cfg)
        with mlflow.start_run(run_name="02_cross_dataset_dedup"):
            mlflow.log_params({
                "minhash_threshold": threshold,
                "exact_only":        str(args.exact_only),
                "num_perm":          _NUM_PERM,
                "min_shingles":      _MIN_SHINGLES,
            })

            log_metrics_dict({
                "n_original":          float(n_orig),
                "n_after_exact_dedup": float(n_after_exact),
                "n_final":             float(n_final),
                "removed_exact":       float(removed_exact),
                "removed_minhash":     float(removed_minhash),
                "total_removed":       float(n_orig - n_final),
                "dedup_rate":          stats["dedup_rate"],
            })

            # Label distribution of deduplicated corpus
            labels_col = table.column("label")
            n_benign    = int(pc.sum(pc.equal(labels_col, 0)).as_py())
            n_malicious = n_final - n_benign
            log_metrics_dict({
                "n_benign":        float(n_benign),
                "n_malicious":     float(n_malicious),
                "imbalance_ratio": round(n_benign / max(1, n_malicious), 3),
            })

            # Per-source record counts
            sources_col = table.column("source").to_pylist()
            source_counts: dict[str, int] = {}
            for s in sources_col:
                source_counts[s] = source_counts.get(s, 0) + 1
            log_metrics_dict({
                f"source_{src}_count": float(cnt)
                for src, cnt in source_counts.items()
            })

            # Per-attack-class record counts
            classes_col = table.column("attack_class").to_pylist()
            class_counts: dict[str, int] = {}
            for c in classes_col:
                class_counts[c] = class_counts.get(c, 0) + 1
            log_metrics_dict({
                f"class_{cls}_count": float(cnt)
                for cls, cnt in class_counts.items()
            })

            mlflow.log_artifact(str(stats_path))
            mlflow.log_artifact(str(samples_path))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


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
    p.add_argument(
        "--force-dedup",
        action="store_true",
        help="Re-run deduplication even if deduped.parquet already exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())