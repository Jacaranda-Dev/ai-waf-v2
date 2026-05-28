"""
stages/4_tokenization/05_compare_tokenizers.py
----------------------------------------------
Stage 3.5 — Canonical side-by-side tokenizer comparison report.

Enhancements over original:
  * Loads pre-computed metrics from scripts 02 and 04 where available,
    avoiding redundant tokenization passes.
  * Falls back to live computation (stratified sample) when cached
    reports are absent.
  * Full metric schema: OOV, fertility, truncation rate, subword-char
    ratio, per-class breakdown, vocab Jaccard overlap.
  * Token shadowing summary from the Track A augmentation report is
    surfaced in the comparison output.
  * Emits a human-readable console summary alongside the JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

from tokenizer_eval import (
    build_comparison_report,
    compute_full_metrics,
    stratified_sample,
)

log = get_logger(__name__)

EVAL_SAMPLE_SIZE = 3_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cached(path: Path) -> dict | None:
    """Return parsed JSON from a cached metrics file, or None if absent."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning(f"Cached metrics file is corrupt, will recompute: {path}")
    return None


def _print_summary(report: dict) -> None:
    """Emit a human-readable comparison table to the log."""
    a = report["track_a"]
    b = report["track_b"]
    w = report["winners"]
    d = report["deltas"]

    lines = [
        "",
        "┌─────────────────────────┬────────────────┬────────────────┬──────────┐",
        "│ Metric                  │    Track A     │    Track B     │  Winner  │",
        "├─────────────────────────┼────────────────┼────────────────┼──────────┤",
    ]

    metrics = [
        ("OOV rate",          "oov_rate"),
        ("Fertility",         "fertility"),
        ("Truncation rate",   "truncation_rate"),
        ("Avg seq len",       "avg_seq_len"),
        ("Subword/char ratio","subword_char_ratio"),
    ]

    for label, key in metrics:
        va   = a.get(key, "N/A")
        vb   = b.get(key, "N/A")
        win  = w.get(key, "—")
        fa   = f"{va:.4f}" if isinstance(va, float) else str(va)
        fb   = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
        lines.append(
            f"│ {label:<23s} │ {fa:>14s} │ {fb:>14s} │ {win:<8s} │"
        )

    lines.append(
        "└─────────────────────────┴────────────────┴────────────────┴──────────┘"
    )

    if report.get("vocab_jaccard_overlap") is not None:
        lines.append(f"  Vocab Jaccard overlap : {report['vocab_jaccard_overlap']:.4f}")

    lines.append(f"  Recommendation        : {report.get('recommendation', '—')}")
    lines.append("")

    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    track_a_dir  = Path(cfg.tokenizer.track_a.output_dir)
    track_b_dir  = Path(cfg.tokenizer.track_b.output_dir)
    reports_dir  = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guard: both tokenizers must exist
    # ------------------------------------------------------------------
    if not (track_a_dir / "tokenizer_config.json").exists():
        log.error("Track A tokenizer missing — run 01_augment_pretrained_vocab.py first.")
        return
    if not (track_b_dir / "tokenizer.json").exists():
        log.error("Track B tokenizer missing — run 03_train_custom_bpe.py first.")
        return

    # ------------------------------------------------------------------
    # Load tokenizers
    # ------------------------------------------------------------------
    tok_a = AutoTokenizer.from_pretrained(str(track_a_dir))
    tok_b = HttpTokenizer.load(str(track_b_dir), cfg.tokenizer.seq_len)
    log.info(
        f"Track A vocab_size={len(tok_a)}  |  "
        f"Track B vocab_size={tok_b.vocab_size}"
    )

    # ------------------------------------------------------------------
    # Attempt to load pre-computed metric caches
    # ------------------------------------------------------------------
    cached_a = _load_cached(reports_dir / "tokenizer_oov_track_a.json")
    cached_b = _load_cached(reports_dir / "tokenizer_oov_track_b.json")

    if cached_a and cached_b:
        log.info(
            "Using pre-computed metrics from scripts 02 & 04 "
            "(set --recompute to override)."
        )
        metrics_a = cached_a
        metrics_b = cached_b
    else:
        # Fall back: live computation on a fresh stratified sample
        log.info(
            "Pre-computed metrics not found — running live evaluation "
            f"(stratified n={EVAL_SAMPLE_SIZE})."
        )
        table  = pq.read_table(
            Path(cfg.paths.data_splits) / "val.parquet",
            columns=["raw", "attack_class"],
        )
        texts  = table["raw"].to_pylist()
        labels = table["attack_class"].to_pylist()

        texts, labels = stratified_sample(
            texts, labels,
            n=min(EVAL_SAMPLE_SIZE, len(texts)),
            seed=cfg.project.seed,
        )
        seq_len  = cfg.tokenizer.seq_len
        metrics_a = compute_full_metrics(tok_a, texts, labels, seq_len, track_name="track_a")
        metrics_b = compute_full_metrics(tok_b, texts, labels, seq_len, track_name="track_b")

    # ------------------------------------------------------------------
    # Inject vocab sets for Jaccard computation
    # ------------------------------------------------------------------
    try:
        metrics_a["_vocab_a"] = list(tok_a.get_vocab().keys())
        metrics_b["_vocab_b"] = list(tok_b.get_vocab().keys()) if hasattr(tok_b, "get_vocab") else []
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build comparison report
    # ------------------------------------------------------------------
    comparison = build_comparison_report(metrics_a, metrics_b)

    # Attach shadowing summary from script 01 if available
    shadow_path = reports_dir / "tokenizer_track_a.json"
    if shadow_path.exists():
        try:
            shadow_data = json.loads(shadow_path.read_text())
            comparison["track_a_shadowing"] = shadow_data.get("shadowing", {}).get("summary")
        except Exception:
            pass

    _print_summary(comparison)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out = reports_dir / "tokenizer_comparison.json"
    out.write_text(json.dumps(comparison, indent=2))
    log.info(f"Full comparison report saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="05_compare_tokenizers"):
            mlflow.log_params({
                "eval_sample_size":  EVAL_SAMPLE_SIZE,
                "seq_len":           cfg.tokenizer.seq_len,
                "used_cache":        bool(cached_a and cached_b),
                "recommendation":    comparison.get("recommendation", ""),
            })
            metrics: dict[str, float] = {}
            for track in ("track_a", "track_b"):
                t = comparison.get(track, {})
                for key in ("oov_rate", "fertility", "truncation_rate", "avg_seq_len"):
                    val = t.get(key)
                    if isinstance(val, (int, float)):
                        metrics[f"{track}_{key}"] = float(val)
            if comparison.get("vocab_jaccard_overlap") is not None:
                metrics["vocab_jaccard_overlap"] = float(comparison["vocab_jaccard_overlap"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side tokenizer comparison.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached metric files and recompute from scratch.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())