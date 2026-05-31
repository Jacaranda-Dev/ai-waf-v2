"""
stages/4_tokenization/04_measure_oov_track_b.py
------------------------------------------------
Stage 3.4 — Evaluate OOV rate, fertility, and truncation for Track B.

Enhancements over original:
  * Stratified sampling replaces sequential [:5000] slicing.
  * Truncation rate and subword-char ratio added via shared module.
  * Deduplicates logic from script 03's embedded evaluation block —
    this script is the canonical Track B evaluation step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

from tokenizer_eval import compute_full_metrics, stratified_sample

log = get_logger(__name__)

EVAL_SAMPLE_SIZE = 5_000


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    require_inputs({
        f"{cfg.tokenizer.track_b.output_dir}/tokenizer.json": "run 03_train_custom_bpe.py",
        "data/splits/val.parquet": "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "tokenizer_oov_track_b.json",
        args.force, "Stage 4.4 Track B OOV measurement"
    ):
        return

    # ------------------------------------------------------------------
    # Load Track B tokenizer
    # ------------------------------------------------------------------
    try:
        tokenizer = HttpTokenizer.load(
            cfg.tokenizer.track_b.output_dir,
            cfg.tokenizer.seq_len,
        )
    except FileNotFoundError:
        log.error(
            "Track B tokenizer not found. "
            "Run 03_train_custom_bpe.py first."
        )
        return

    log.info(f"Loaded Track B tokenizer (vocab_size={tokenizer.vocab_size})")
    timer = StepTimer()

    # ------------------------------------------------------------------
    # Load validation data — full table, then stratified-sample
    # ------------------------------------------------------------------
    val_path = Path(cfg.paths.data_splits) / "val.parquet"
    if not val_path.exists():
        log.error("Validation split not found at '%s'.", val_path)
        return

    table  = pq.read_table(val_path, columns=["raw", "attack_class"])
    texts  = table["raw"].to_pylist()
    labels = table["attack_class"].to_pylist()

    texts, labels = stratified_sample(
        texts, labels,
        n=min(EVAL_SAMPLE_SIZE, len(texts)),
        seed=cfg.project.seed,
    )
    log.info(
        f"Stratified sample: {len(texts)} requests across "
        f"{len(set(labels))} attack classes"
    )

    # ------------------------------------------------------------------
    # Compute unified metrics
    # ------------------------------------------------------------------
    seq_len = cfg.tokenizer.seq_len
    with timer.step("compute_metrics"):
        result  = compute_full_metrics(
            tokenizer, texts, labels,
            seq_len=seq_len,
            unk_token="[UNK]",
            track_name="track_b",
        )
    result["vocab_size"] = tokenizer.vocab_size

    log.info(
        f"Track B | oov={result['oov_rate']:.4f}  "
        f"fertility={result['fertility']:.4f}  "
        f"truncation={result['truncation_rate']:.4f}  "
        f"subword_char_ratio={result['subword_char_ratio']:.2f}"
    )
    for cls, m in result["per_class"].items():
        log.info(
            f"  {cls:<22s} oov={m['oov_rate']:.4f}  "
            f"fertility={m['fertility']:.4f}  "
            f"trunc={m['truncation_rate']:.4f}  n={m['n_samples']}"
        )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out = Path(cfg.paths.reports) / "metrics" / "tokenizer_oov_track_b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result["timings_s"] = timer.timings
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Track B evaluation report saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="04_measure_oov_track_b"):
            mlflow.log_params({
                "eval_sample_size": EVAL_SAMPLE_SIZE,
                "seq_len":          cfg.tokenizer.seq_len,
                "tokenizer_dir":    cfg.tokenizer.track_b.output_dir,
                "vocab_size":       tokenizer.vocab_size,
            })
            log_metrics_dict({
                "oov_rate":           float(result["oov_rate"]),
                "fertility":          float(result["fertility"]),
                "truncation_rate":    float(result["truncation_rate"]),
                "avg_seq_len":        float(result["avg_seq_len"]),
                "subword_char_ratio": float(result["subword_char_ratio"]),
            })
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Track B tokenizer metrics.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())