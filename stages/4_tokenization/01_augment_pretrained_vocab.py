"""
stages/4_tokenization/01_augment_pretrained_vocab.py
-----------------------------------------------------
Stage 3.1 — Augment BERT-base vocab with HTTP-specific tokens (Track A).

Enhancements over original:
  * Token shadowing analysis: verifies that added tokens are treated as
    atomic units and not fragmented by the underlying WordPiece scorer.
  * Shadow report is written to tokenizer_track_a.json alongside standard
    augmentation stats so downstream consumers can surface warnings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_waf_v2.tokenizer.vocab_utils import augment_pretrained_vocab
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

from tokenizer_eval import check_token_shadowing, summarise_shadowing

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg    = load_config(args.config)
    tok_a  = cfg.tokenizer.track_a

    # ------------------------------------------------------------------
    # Augment vocabulary
    # ------------------------------------------------------------------
    tokenizer, n_added = augment_pretrained_vocab(
        base_model=tok_a.base_model,
        new_tokens=tok_a.http_tokens,
        output_dir=tok_a.output_dir,
    )
    log.info(f"Track A: {n_added} new tokens added to '{tok_a.base_model}' vocab")
    log.info(f"New vocab size: {len(tokenizer)}")

    # ------------------------------------------------------------------
    # Shadowing analysis
    # After augmentation, WordPiece may still prefer to fragment tokens
    # whose subword decomposition scores higher than the newly added
    # whole-token entry (e.g. "UNION SELECT" → "UNI", "##ON", …).
    # ------------------------------------------------------------------
    log.info("Running token shadowing analysis...")
    shadow_report = check_token_shadowing(tokenizer, tok_a.http_tokens)
    shadow_summary = summarise_shadowing(shadow_report)

    if shadow_summary["n_shadowed"] > 0:
        log.warning(
            f"Shadowing detected: {shadow_summary['n_shadowed']}/{shadow_summary['total_checked']} "
            f"tokens are still fragmented → {shadow_summary['shadowed_tokens']}"
        )
        log.warning(
            "Consider using the 'add_prefix_space' option or post-processing the tokenizer "
            "config to force atomic encoding of these tokens."
        )
    else:
        log.info(
            f"No shadowing detected: all {shadow_summary['total_checked']} tokens "
            "are encoded as single units. ✓"
        )

    # ------------------------------------------------------------------
    # Persist report
    # ------------------------------------------------------------------
    stats = {
        "base_model":      tok_a.base_model,
        "n_added":         n_added,
        "new_vocab_size":  len(tokenizer),
        "tokens_added":    tok_a.http_tokens,
        "shadowing": {
            "summary": shadow_summary,
            "detail":  shadow_report,
        },
    }

    out = Path(cfg.paths.reports) / "metrics" / "tokenizer_track_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    log.info(f"Track A augmentation report saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_augment_pretrained_vocab"):
            mlflow.log_params({
                "base_model":     tok_a.base_model,
                "n_http_tokens":  len(tok_a.http_tokens),
            })
            log_metrics_dict({
                "n_tokens_added":    float(n_added),
                "new_vocab_size":    float(len(tokenizer)),
                "n_shadowed":        float(shadow_summary.get("n_shadowed", 0)),
                "total_checked":     float(shadow_summary.get("total_checked", 0)),
            })
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment BERT vocab with HTTP tokens (Track A).")
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())