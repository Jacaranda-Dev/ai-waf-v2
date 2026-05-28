"""
stages/3_data_augmentation/11_quality_gate.py
---------------------------------------------
Unified Quality Pipeline  (merges 11_format_validation, 12_tokenizer_coverage_check,
13_semantic_dedup, 14_label_consistency)

Four passes run in sequence, all controlled by a single Pipeline class:

  Pass 1 — FormatValidator    : valid HTTP structure, length bounds
  Pass 2 — TokenizerCoverage  : reject if UNK-rate > threshold
  Pass 3 — SemanticDedup      : MinHash LSH at configurable Jaccard threshold
  Pass 4 — LabelConsistency   : heuristic aligned with ModSecurity CRS baseline
                                (same regex set as ai_waf_v2.baselines.crs_heuristic)

Cross-augmentation leakage check: after filtering, verify no augmented record
has Jaccard > 0.70 with test.parquet or canary.parquet (hard requirement).

Enhancements over original scripts:
  - ModSecurity-aligned heuristic (replaces divergent standalone regex)
  - FormatValidator is stateless and runs in parallel across Parquet partitions
  - Conflict-flagged benign edge-cases are kept but marked; pure conflicts are quarantined
  - Leakage guard against test / canary splits

Run:
    python stages/3_data_augmentation/11_quality_gate.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from ai_waf_v2.data.schema import HttpRecord, PARQUET_SCHEMA
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ModSecurity CRS-aligned heuristics
# (Must match the patterns used in ai_waf_v2.baselines.crs_heuristic to avoid
#  the divergence that existed between 11_format_validation.py and 14_label_consistency.py)
# ─────────────────────────────────────────────────────────────────────────────

class CRSHeuristic:
    """
    Minimal ModSecurity CRS rule set implemented as compiled regex patterns.
    Uses the same groupings as the CRS paranoia level 1 defaults so that
    the quality gate's consistency check is directly comparable to the
    baseline evaluation metrics produced by Stage 2.
    """

    _SQLI = re.compile(
        r"(UNION\s+(?:ALL\s+)?SELECT|SELECT\s+.+\s+FROM|"
        r"DROP\s+TABLE|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM|"
        r"OR\s+1\s*=\s*1|AND\s+1\s*=\s*1|OR\s+'[^']+'='[^']+'|"
        r"SLEEP\s*\(|BENCHMARK\s*\(|WAITFOR\s+DELAY|"
        r"EXTRACTVALUE\s*\(|UPDATEXML\s*\(|"
        r"information_schema|mysql\.user|pg_sleep)",
        re.IGNORECASE,
    )
    _XSS = re.compile(
        r"(<\s*script|onerror\s*=|onload\s*=|onfocus\s*=|ontoggle\s*=|"
        r"javascript\s*:|data\s*:\s*text/html|<\s*iframe|alert\s*\(|"
        r"document\.cookie|<\s*svg\s+on)",
        re.IGNORECASE,
    )
    _LFI = re.compile(
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.\.%2f|%252e%252e|"
        r"php://filter|php://input|data://text|/etc/passwd|/etc/shadow|"
        r"\\windows\\system32)",
        re.IGNORECASE,
    )
    _SSRF = re.compile(
        r"(169\.254\.169\.254|metadata\.google\.internal|"
        r"localhost(?::\d+)?/admin|127\.0\.0\.1/|0\.0\.0\.0/|"
        r"\[::1\]/|dict://|gopher://)",
        re.IGNORECASE,
    )
    _CMDI = re.compile(
        r"(;\s*(?:id|whoami|cat\s|ls\s|uname|ps\s|env\b)|"
        r"\|\s*(?:id|whoami|cat\s|ls\s|uname)|"
        r"`(?:id|whoami|cat\s|ls\s)`|\$\((?:id|whoami|ls\s)|"
        r"bash\s+-i\s+>&|nc\s+-e\s+/bin)",
        re.IGNORECASE,
    )

    PATTERNS = {
        "sqli": _SQLI,
        "xss":  _XSS,
        "lfi":  _LFI,
        "ssrf": _SSRF,
        "cmdi": _CMDI,
    }

    @classmethod
    def matches(cls, text: str) -> set[str]:
        """Return set of attack class names matched by CRS heuristics."""
        return {name for name, pat in cls.PATTERNS.items() if pat.search(text)}

    @classmethod
    def any_match(cls, text: str) -> bool:
        return bool(cls.matches(text))


CRS = CRSHeuristic()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline passes  (stateless — each accepts an HttpRecord, returns pass/fail)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    passed:  bool
    reason:  str = ""


class FormatValidator:
    """Stateless HTTP structure validator — safe for concurrent execution."""

    _METHOD_RE = re.compile(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s")

    def __call__(self, record: HttpRecord) -> FilterResult:
        raw = record.raw or ""
        if len(raw) < 10:
            return FilterResult(False, "format:too_short")
        if not self._METHOD_RE.match(raw):
            return FilterResult(False, "format:invalid_method")
        if len(raw) > 8_192:
            return FilterResult(False, "format:too_long")
        return FilterResult(True)


class TokenizerCoverage:
    """Rejects records whose UNK-token rate exceeds the configured threshold."""

    def __init__(self, tokenizer, max_unk_ratio: float, pad_id: int, unk_id: int):
        self.tokenizer     = tokenizer
        self.max_unk_ratio = max_unk_ratio
        self.pad_id        = pad_id
        self.unk_id        = unk_id

    def __call__(self, record: HttpRecord) -> FilterResult:
        enc  = self.tokenizer.encode(record.raw)
        ids  = [i for i in enc.ids if i != self.pad_id]
        rate = sum(1 for i in ids if i == self.unk_id) / max(1, len(ids))
        if rate > self.max_unk_ratio:
            return FilterResult(False, f"high_unk_rate:{rate:.3f}")
        return FilterResult(True)


class SemanticDedup:
    """
    Stateful MinHash LSH deduplicator.
    Maintains internal state across calls — not thread-safe; run single-threaded.
    """

    def __init__(self, threshold: float = 0.90, num_perm: int = 64):
        self.lsh      = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._seen: set[str] = set()

    def _minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for i in range(max(1, len(text) - 2)):
            m.update(text[i:i+3].encode("utf-8", errors="replace"))
        return m

    def __call__(self, record: HttpRecord) -> FilterResult:
        mh = self._minhash(record.raw)
        if self.lsh.query(mh):
            return FilterResult(False, "semantic_duplicate")
        self.lsh.insert(record.id, mh)
        return FilterResult(True)

    def is_near_duplicate_of(self, text: str) -> bool:
        """Check a single string against the existing LSH index (used for leakage guard)."""
        mh = self._minhash(text)
        return bool(self.lsh.query(mh))


class LabelConsistency:
    """
    Compares assigned label against CRS heuristic signal.
    Policy (aligned with CRS baseline):
      - label=1 AND no CRS match: flag as potential false-positive seed (kept, source tagged)
      - label=0 AND CRS matches AND source is NOT 'benign_edge_case': quarantine
      - label=0 AND CRS matches AND source is 'benign_edge_case': keep, flag only
    """

    def __call__(self, record: HttpRecord) -> tuple[FilterResult, str]:
        crs_hits = CRS.matches(record.raw)
        is_edge  = "edge" in record.source

        if record.label == 0 and crs_hits and not is_edge:
            return FilterResult(False, f"label_conflict:benign_with_crs_hit:{','.join(crs_hits)}"), "quarantine"

        if record.label == 0 and crs_hits and is_edge:
            return FilterResult(True, "conflict_flagged"), "flag"

        if record.label == 1 and not crs_hits:
            return FilterResult(True, "undetected_by_crs"), "flag"

        return FilterResult(True), "pass"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-augmentation leakage guard
# ─────────────────────────────────────────────────────────────────────────────

def _build_holdout_lsh(holdout_paths: list[Path], threshold: float = 0.70) -> MinHashLSH:
    """Build a MinHash LSH index from test + canary splits for leakage detection."""
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    n   = 0
    for p in holdout_paths:
        if not p.exists():
            continue
        for row in pq.read_table(p, columns=["id", "raw"]).to_pylist():
            mh = MinHash(num_perm=64)
            raw = row.get("raw") or ""
            for i in range(max(1, len(raw) - 2)):
                mh.update(raw[i:i+3].encode("utf-8", errors="replace"))
            try:
                lsh.insert(row["id"], mh)
                n += 1
            except Exception:
                pass
    log.info(f"Leakage guard: indexed {n:,} holdout records from {[p.name for p in holdout_paths if p.exists()]}")
    return lsh


def _leakage_check(records: list[HttpRecord], holdout_lsh: MinHashLSH, num_perm: int = 64) -> tuple[list[HttpRecord], int]:
    """Remove records that are near-duplicates of holdout set. Returns (clean, n_removed)."""
    clean   = []
    removed = 0
    for r in records:
        mh = MinHash(num_perm=num_perm)
        for i in range(max(1, len(r.raw) - 2)):
            mh.update(r.raw[i:i+3].encode("utf-8", errors="replace"))
        if holdout_lsh.query(mh):
            removed += 1
        else:
            clean.append(r)
    return clean, removed


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QualityPipeline:

    def __init__(
        self,
        format_val:    FormatValidator,
        tok_coverage:  TokenizerCoverage | None,
        dedup:         SemanticDedup,
        label_check:   LabelConsistency,
        holdout_lsh:   MinHashLSH | None,
    ):
        self.format_val   = format_val
        self.tok_coverage = tok_coverage
        self.dedup        = dedup
        self.label_check  = label_check
        self.holdout_lsh  = holdout_lsh

    def run(
        self,
        records:    list[HttpRecord],
        max_workers: int = 4,
    ) -> tuple[list[HttpRecord], list[tuple[HttpRecord, str]]]:
        """
        Execute all passes.
        Pass 1 (format) runs in parallel.
        Pass 2 (tokenizer) runs in parallel.
        Pass 3 (dedup) and Pass 4 (label) run single-threaded (stateful / policy).
        """
        passed:   list[HttpRecord]              = []
        rejected: list[tuple[HttpRecord, str]]  = []

        # ── Pass 1: format validation (parallel) ──
        log.info(f"Pass 1/4 — HTTP format validation ({len(records):,} records)...")
        after_fmt: list[HttpRecord] = []
        fmt_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.format_val, r): r for r in records}
            for fut in as_completed(futures):
                r   = futures[fut]
                res = fut.result()
                if res.passed:
                    after_fmt.append(r)
                else:
                    rejected.append((r, res.reason))
        log.info(f"  ✓ {len(after_fmt):,}  ✗ {len(records) - len(after_fmt):,}")

        # ── Pass 2: tokenizer UNK rate (parallel) ──
        log.info("Pass 2/4 — Tokenizer coverage check...")
        after_tok: list[HttpRecord] = []
        if self.tok_coverage:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.tok_coverage, r): r for r in after_fmt}
                for fut in as_completed(futures):
                    r   = futures[fut]
                    res = fut.result()
                    if res.passed:
                        after_tok.append(r)
                    else:
                        rejected.append((r, res.reason))
        else:
            after_tok = after_fmt
            log.info("  (skipped — tokenizer not available)")
        log.info(f"  ✓ {len(after_tok):,}  ✗ {len(after_fmt) - len(after_tok):,}")

        # ── Pass 3: semantic dedup (sequential — stateful LSH) ──
        log.info("Pass 3/4 — Semantic deduplication (MinHash LSH)...")
        after_dedup: list[HttpRecord] = []
        for r in after_tok:
            res = self.dedup(r)
            if res.passed:
                after_dedup.append(r)
            else:
                rejected.append((r, res.reason))
        log.info(f"  ✓ {len(after_dedup):,}  ✗ {len(after_tok) - len(after_dedup):,}")

        # ── Pass 4: label consistency (sequential — policy) ──
        log.info("Pass 4/4 — CRS-aligned label consistency check...")
        n_flagged = n_quarantined = 0
        after_label: list[HttpRecord] = []
        for r in after_dedup:
            res, action = self.label_check(r)
            if action == "quarantine":
                rejected.append((r, res.reason))
                n_quarantined += 1
            elif action == "flag":
                # Tag source but keep in training set
                object.__setattr__(r, "source", r.source + "_crs_flagged")
                after_label.append(r)
                n_flagged += 1
            else:
                after_label.append(r)
        log.info(f"  ✓ {len(after_label):,}  quarantined={n_quarantined:,}  flagged(kept)={n_flagged:,}")

        # ── Leakage guard ──────────────────────────────────────────────────
        if self.holdout_lsh:
            log.info("Leakage guard — checking proximity to test/canary splits...")
            after_label, n_leaked = _leakage_check(after_label, self.holdout_lsh)
            log.info(f"  Removed {n_leaked:,} records too similar to holdout splits")

        passed = after_label
        return passed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _load_all_augmented(aug_dir: Path, base_deduped: Path) -> list[HttpRecord]:
    parquet_files = list(aug_dir.rglob("*.parquet"))
    if base_deduped.exists():
        parquet_files.append(base_deduped)

    dfs = []
    for fp in parquet_files:
        try:
            dfs.append(pq.read_table(fp).to_pandas())
        except Exception as e:
            log.warning(f"Could not load {fp}: {e}")

    if not dfs:
        return []

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["id"])
    return [HttpRecord.from_dict(row) for _, row in df.iterrows()]


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    filt_cfg = cfg.augmentation.filtering

    # ── Load tokenizer ────────────────────────────────────────────────────
    tok_pass = None
    try:
        from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
        tok = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
        unk_id = tok.token_to_id("[UNK]") or 1
        pad_id = tok.pad_token_id
        tok_pass = TokenizerCoverage(tok, filt_cfg.max_unk_ratio, pad_id, unk_id)
        log.info("Track B tokenizer loaded for UNK coverage pass")
    except FileNotFoundError:
        log.warning("Track B tokenizer not found — UNK coverage pass will be skipped")

    # ── Load data ─────────────────────────────────────────────────────────
    aug_dir   = Path(cfg.paths.data_augmented)
    base_path = Path(cfg.paths.data_normalized) / "deduped.parquet"
    records   = _load_all_augmented(aug_dir, base_path)
    if not records:
        log.error("No augmented records found — run Modules A, B, C first")
        return
    log.info(f"Loaded {len(records):,} total records for quality gating")

    # ── Build holdout LSH for leakage guard ───────────────────────────────
    splits_dir = Path(cfg.paths.data_splits)
    holdout_paths = [splits_dir / "test.parquet", splits_dir / "canary.parquet"]
    holdout_lsh = None
    if any(p.exists() for p in holdout_paths):
        holdout_lsh = _build_holdout_lsh(holdout_paths, threshold=0.70)

    # ── Build pipeline ────────────────────────────────────────────────────
    pipeline = QualityPipeline(
        format_val   = FormatValidator(),
        tok_coverage = tok_pass,
        dedup        = SemanticDedup(threshold=cfg.data.dedup.semantic_threshold),
        label_check  = LabelConsistency(),
        holdout_lsh  = holdout_lsh,
    )

    passed, rejected = pipeline.run(records, max_workers=args.workers)

    # ── Write outputs ─────────────────────────────────────────────────────
    filtered_dir = Path(cfg.paths.data_filtered)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    def _write(recs: list[HttpRecord], name: str):
        if not recs:
            return
        df = pd.DataFrame([r.model_dump() for r in recs])
        pq.write_table(
            pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False),
            filtered_dir / name,
            compression="snappy",
        )

    _write(passed,                 "filtered.parquet")
    _write([r for r, _ in rejected], "rejected.parquet")

    n_total    = len(records)
    n_passed   = len(passed)
    n_rejected = len(rejected)
    rej_rate   = n_rejected / max(1, n_total)

    log.info(
        f"\nQuality gate complete:\n"
        f"  Input:    {n_total:,}\n"
        f"  Passed:   {n_passed:,}\n"
        f"  Rejected: {n_rejected:,} ({rej_rate*100:.1f}%)"
    )

    def _count_reasons(rejected_list):
        counts: dict[str, int] = {}
        for _, reason in rejected_list:
            top = reason.split(":")[0]
            counts[top] = counts.get(top, 0) + 1
        return counts

    stats = {
        "n_input":           n_total,
        "n_passed":          n_passed,
        "n_rejected":        n_rejected,
        "rejection_rate":    round(rej_rate, 4),
        "rejection_reasons": _count_reasons(rejected),
        "crs_heuristic":     "ModSecurity CRS aligned (paranoia level 1)",
        "leakage_threshold": 0.70,
    }
    sp = Path(cfg.paths.reports) / "metrics" / "quality_gate.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stats, indent=2))

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="04_quality_gate"):
            mlflow.log_params({
                "max_unk_ratio":       filt_cfg.max_unk_ratio,
                "semantic_threshold":  cfg.data.dedup.semantic_threshold,
                "leakage_threshold":   0.70,
                "n_workers":           args.workers,
            })
            metrics: dict[str, float] = {
                "n_input":        float(n_total),
                "n_passed":       float(n_passed),
                "n_rejected":     float(n_rejected),
                "rejection_rate": float(rej_rate),
            }
            for reason, count in stats.get("rejection_reasons", {}).items():
                metrics[f"rejected_{reason}"] = float(count)
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(sp))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/pipeline.yaml")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for format/tokenizer passes")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())