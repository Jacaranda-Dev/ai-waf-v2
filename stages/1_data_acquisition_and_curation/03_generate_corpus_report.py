"""
stages/1_data_acquisition_and_curation/03_generate_corpus_report.py
--------------------------------------------------------------------
Stage 4 — Single-pass corpus health report.

Replaces five separate scripts that each loaded the same Parquet file:
    04_dataset_analysis.py      → dataset_analysis.json
    05_datasheet.py             → datasheet.json
    06_taxonomy_coverage.py     → taxonomy_coverage.json
    07_length_distribution.py   → length_distribution.json
    08_taxonomy_inventory.py    → taxonomy_inventory.json

All five JSON artefacts are still produced at the same paths so downstream
consumers (CI checks, dashboards, augmentation targeting) are unaffected.
The Parquet file is read exactly once; all metrics are computed from that
single in-memory DataFrame before anything is written to disk.

Structure
─────────
  _load_corpus()             read deduped.parquet (with fallback)
  _report_dataset_analysis() counts, imbalance, method distribution
  _report_taxonomy_inventory() per-class status, missing / low-count
  _report_taxonomy_coverage()  class × source coverage matrix
  _report_length_distribution() char / token length percentiles
  _report_datasheet()         Gebru et al. metadata stub
  run()                      orchestrates all five in order

Run:
    python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py \\
        --config config/pipeline.yaml

    # write reports without re-reading Parquet if you want to re-run
    # individual sections, pass --section (can be repeated):
    python ... --section taxonomy_inventory --section length_distribution
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# All section keys — used to validate --section args and control execution.
ALL_SECTIONS = frozenset({
    "dataset_analysis",
    "taxonomy_inventory",
    "taxonomy_coverage",
    "length_distribution",
    "datasheet",
})


# ─────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────

def _load_corpus(cfg) -> pd.DataFrame:
    """
    Load the canonical corpus in priority order:
      1. deduped.parquet   (preferred — output of Stage 03)
      2. all_datasets.parquet (fallback if dedup hasn't run yet)

    Raises FileNotFoundError if neither exists.
    """
    candidates = [
        Path(cfg.paths.data_normalized) / "deduped.parquet",
        Path(cfg.paths.data_normalized) / "all_datasets.parquet",
    ]
    for path in candidates:
        if path.exists():
            log.info(f"Loading corpus from {path.name}…")
            df = pq.read_table(path).to_pandas()
            log.info(f"  {len(df):,} records loaded")
            return df

    raise FileNotFoundError(
        f"No corpus found. Expected one of: {[str(p) for p in candidates]}. "
        "Run Stage 01 (acquire + normalise) and Stage 03 (dedup) first."
    )


# ─────────────────────────────────────────────────────────
# Section: dataset_analysis  (replaces 04_dataset_analysis.py)
# ─────────────────────────────────────────────────────────

def _report_dataset_analysis(df: pd.DataFrame, reports_dir: Path) -> dict[str, Any]:
    """
    Compute and persist overall corpus statistics.

    Metrics produced:
      total_records, benign / malicious counts, imbalance_ratio,
      source_counts, attack_class_counts, raw_length percentiles, method distribution.
    """
    log.info("Section: dataset_analysis")

    raw_lens = df["raw"].str.len().values

    stats: dict[str, Any] = {
        "total_records":       int(len(df)),
        "benign":              int((df["label"] == 0).sum()),
        "malicious":           int((df["label"] == 1).sum()),
        "imbalance_ratio":     round(
            float((df["label"] == 0).sum()) / max(1, float((df["label"] == 1).sum())), 3
        ),
        "source_counts":       df["source"].value_counts().to_dict(),
        "attack_class_counts": df["attack_class"].value_counts().to_dict(),
        "raw_length": {
            "min":    int(raw_lens.min()),
            "max":    int(raw_lens.max()),
            "mean":   round(float(raw_lens.mean()), 1),
            "median": round(float(np.median(raw_lens)), 1),
            "p95":    round(float(np.percentile(raw_lens, 95)), 1),
            "p99":    round(float(np.percentile(raw_lens, 99)), 1),
        },
        "methods": df["method"].value_counts().to_dict(),
    }

    out = reports_dir / "dataset_analysis.json"
    out.write_text(json.dumps(stats, indent=2))

    log.info(
        f"  total={stats['total_records']:,}  "
        f"benign={stats['benign']:,}  malicious={stats['malicious']:,}  "
        f"imbalance={stats['imbalance_ratio']:.2f}  "
        f"p99_len={stats['raw_length']['p99']}"
    )
    log.info(f"  → {out}")
    return stats


# ─────────────────────────────────────────────────────────
# Section: taxonomy_inventory  (replaces 08_taxonomy_inventory.py)
# ─────────────────────────────────────────────────────────

def _report_taxonomy_inventory(
    df: pd.DataFrame,
    cfg,
    reports_dir: Path,
) -> dict[str, Any]:
    """
    Enumerate attack-class coverage and flag gaps for augmentation targeting.

    Output taxonomy_inventory.json — consumed by the augmentation stage to
    decide which classes need synthetic data generation.
    """
    log.info("Section: taxonomy_inventory")

    min_samples = cfg.data.augmentation.min_samples_per_class  


    malicious       = df[df["label"] == 1]
    class_counts    = Counter(df["attack_class"].tolist())
    source_counts   = Counter(df["source"].tolist())

    required  = set(cfg.data.schema.attack_classes)
    present   = {cls for cls, cnt in class_counts.items() if cls != "benign" and cnt > 0}
    missing   = required - present
    low_count = {
        cls: cnt for cls, cnt in class_counts.items()
        if cls != "benign" and 0 < cnt < min_samples
    }

    # Human-readable table in the log
    log.info(f"  {'Class':25s}  {'Count':>8}  {'Status':>8}")
    log.info(f"  {'-'*46}")
    for cls in sorted(required | present):
        cnt    = class_counts.get(cls, 0)
        status = "OK" if cnt >= min_samples else ("LOW" if cnt > 0 else "MISSING")
        log.info(f"  {cls:25s}  {cnt:>8,}  {status:>8}")
    log.info(f"  Benign: {class_counts.get('benign', 0):,}")

    if missing:
        log.warning(f"  Classes with ZERO samples (need augmentation): {sorted(missing)}")
    if low_count:
        log.warning(f"  Low-count classes: {dict(sorted(low_count.items()))}")

    result: dict[str, Any] = {
        "class_counts":      dict(class_counts),
        "source_counts":     dict(source_counts),
        "required_classes":  sorted(required),
        "present_classes":   sorted(present),
        "missing_classes":   sorted(missing),
        "low_count_classes": low_count,
        "total_samples":     int(len(df)),
        "benign":            int((df["label"] == 0).sum()),
        "malicious":         int((df["label"] == 1).sum()),
    }

    out = reports_dir / "taxonomy_inventory.json"
    out.write_text(json.dumps(result, indent=2))
    log.info(f"  → {out}")
    return result


# ─────────────────────────────────────────────────────────
# Section: taxonomy_coverage  (replaces 05/06 taxonomy logic)
# ─────────────────────────────────────────────────────────

def _report_taxonomy_coverage(
    df: pd.DataFrame,
    cfg,
    reports_dir: Path,
) -> dict[str, Any]:
    """
    Build a class × source coverage matrix for malicious records only.

    Answers: "For each attack class, which source datasets contribute samples,
    and how many?"  Used to identify dataset-specific coverage gaps.
    """
    log.info("Section: taxonomy_coverage")

    malicious = df[df["label"] == 1]
    required  = set(cfg.data.schema.attack_classes)
    present   = set(malicious["attack_class"].unique())

    # Build matrix: source → {attack_class: count}
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in malicious.iterrows():
        matrix[row["source"]][row["attack_class"]] += 1

    all_classes = required | present

    result: dict[str, Any] = {
        # Normalise so every source row contains every known class (0 if absent)
        "coverage_matrix": {
            src: {cls: int(matrix[src].get(cls, 0)) for cls in sorted(all_classes)}
            for src in sorted(matrix)
        },
        "missing_classes": sorted(required - present),
        "class_totals":    malicious["attack_class"].value_counts().to_dict(),
    }

    out = reports_dir / "taxonomy_coverage.json"
    out.write_text(json.dumps(result, indent=2))

    if result["missing_classes"]:
        log.warning(f"  Classes with ZERO samples: {result['missing_classes']}")
    log.info(f"  → {out}")
    return result


# ─────────────────────────────────────────────────────────
# Section: length_distribution  (replaces 07_length_distribution.py)
# ─────────────────────────────────────────────────────────

def _report_length_distribution(
    df: pd.DataFrame,
    cfg,
    reports_dir: Path,
) -> dict[str, Any]:
    """
    Compute character and estimated token-length distributions.

    BPE approximation: chars / 4  (empirically reasonable for HTTP payloads).
    The `pct_over_limit` metric tells how many requests exceed the tokenizer
    window and will require truncation.
    """
    log.info("Section: length_distribution")

    chars   = df["raw"].str.len().values
    pcts    = [50, 75, 90, 95, 99, 100]
    seq_len = cfg.tokenizer.seq_len

    char_dist: dict[str, float] = {
        f"p{p}": round(float(np.percentile(chars, p)), 1) for p in pcts
    }
    char_dist["mean"] = round(float(chars.mean()), 1)
    char_dist["min"]  = int(chars.min())

    result: dict[str, Any] = {
        "char_length":                       char_dist,
        "estimated_token_length_bpe_approx": {k: round(v / 4, 1) for k, v in char_dist.items()},
        "seq_len_limit":                     seq_len,
        "pct_requests_over_limit_approx":    round(float((chars > seq_len * 4).mean() * 100), 2),
        "truncation_strategy":               "keep method+path+headers; truncate body last",
    }

    out = reports_dir / "length_distribution.json"
    out.write_text(json.dumps(result, indent=2))

    log.info(
        f"  p50={char_dist['p50']} chars  "
        f"p99={char_dist['p99']} chars  "
        f"~{result['pct_requests_over_limit_approx']}% over seq_len={seq_len}"
    )
    log.info(f"  → {out}")
    return result


# ─────────────────────────────────────────────────────────
# Section: datasheet  (replaces 05_datasheet.py)
# ─────────────────────────────────────────────────────────

def _report_datasheet(cfg, reports_dir: Path) -> dict[str, Any]:
    """
    Write a Gebru et al. (2018) datasheet stub for the WAF-AI corpus.

    This section does not require the DataFrame — it reads only from config.
    It is included here so all curation metadata is produced in one invocation.
    """
    log.info("Section: datasheet")

    datasheet: dict[str, Any] = {
        "dataset_name": "WAF-AI HTTP Classification Dataset",
        "version":      cfg.project.version,
        "motivation": {
            "purpose":  "Train and evaluate transformer-based WAF classifiers",
            "creators": "WAF-AI research project",
            "funding":  "NSF CyberAI / SFS program",
        },
        "composition": {
            "instances":   "HTTP/1.1 request records",
            "labels":      {"0": "benign", "1": "malicious"},
            "attack_classes":        cfg.data.schema.attack_classes,
            "sources":               [d.name for d in cfg.data.datasets],
            "augmentation_methods": [
                "rule-based mutations",
                "grammar-based generation",
                "security-LLM payloads",
                "cloud-LLM framing",
            ],
        },
        "collection": {
            "methods":   "public datasets + synthetic augmentation",
            "timeframe": "2007–2025 (original datasets)",
            "consent":   "public research datasets only",
            "pii":       "none — all synthetic or anonymised",
        },
        "preprocessing": [
            "schema normalisation (Stage 01)",
            "cross-dataset deduplication: exact SHA-256 + MinHash LSH (Stage 03)",
            "format validation (HTTP structure check)",
            "tokenizer coverage filtering (UNK rate < 15%)",
            "stratified train/val/test/adversarial/canary split (Stage 05)",
        ],
        "uses": {
            "suitable": [
                "Training WAF binary classifiers",
                "Evaluating adversarial robustness of HTTP classifiers",
                "Benchmarking tokenization strategies for HTTP",
            ],
            "not_suitable": [
                "Direct production deployment without further validation",
                "Legal or forensic attribution of attacks",
            ],
        },
        "distribution": "research use only",
        "maintenance":  "maintained by project authors",
    }

    out = reports_dir / "datasheet.json"
    out.write_text(json.dumps(datasheet, indent=2))
    log.info(f"  → {out}")
    return datasheet


# ─────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine which sections to run
    sections = set(args.section) if args.section else ALL_SECTIONS
    unknown  = sections - ALL_SECTIONS
    if unknown:
        log.error(f"Unknown section(s): {sorted(unknown)}. Valid: {sorted(ALL_SECTIONS)}")
        return

    # ── Single Parquet read ───────────────────────────────────────────────────
    # All sections that need the DataFrame share this one load.
    # _report_datasheet() reads only config and is called unconditionally
    # without touching `df`.
    needs_df = sections - {"datasheet"}
    df: pd.DataFrame | None = None

    if needs_df:
        try:
            df = _load_corpus(cfg)
        except FileNotFoundError as exc:
            log.error(str(exc))
            return

    # ── Run each requested section ────────────────────────────────────────────
    log.info(f"\nGenerating corpus report — sections: {sorted(sections)}")
    log.info("─" * 60)

    if "dataset_analysis" in sections and df is not None:
        _report_dataset_analysis(df, reports_dir)

    if "taxonomy_inventory" in sections and df is not None:
        _report_taxonomy_inventory(df, cfg, reports_dir)

    if "taxonomy_coverage" in sections and df is not None:
        _report_taxonomy_coverage(df, cfg, reports_dir)

    if "length_distribution" in sections and df is not None:
        _report_length_distribution(df, cfg, reports_dir)

    if "datasheet" in sections:
        _report_datasheet(cfg, reports_dir)

    log.info("─" * 60)
    log.info(f"Corpus report complete. All artefacts written to {reports_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4 — Single-pass corpus health report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument(
        "--section",
        nargs="+",
        metavar="SECTION",
        choices=sorted(ALL_SECTIONS),
        default=None,
        help=(
            "Run only the specified section(s). "
            f"Choices: {sorted(ALL_SECTIONS)}. "
            "Default: all sections."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())