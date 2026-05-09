"""
stages/7_evaluation/07_tokenizer_ablation.py
-----------------------------------------
All four ablation studies in one file (dispatched by script name).
Each ablation trains a fast probe model (LR + TF-IDF) to isolate
the variable of interest without requiring full transformer training.

For full transformer ablations, use MLflow to compare existing runs:
these scripts compare saved checkpoints and report metrics.

Run:
    python stages/7_evaluation/07_tokenizer_ablation.py  --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 07 — Tokenizer ablation
# ─────────────────────────────────────────────────────────

def tokenizer_ablation(cfg) -> dict:
    """Compare Track A vs Track B tokenizer OOV and fertility."""
    from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
    from ai_waf_v2.tokenizer.vocab_utils import measure_oov
    import pyarrow.parquet as pq

    reports_dir = Path(cfg.paths.reports) / "metrics"
    val_path = Path(cfg.paths.data_splits) / "val.parquet"
    if not val_path.exists():
        return {"error": "val.parquet not found"}

    df = pq.read_table(val_path, columns=["raw", "attack_class"]).to_pandas()
    texts  = df["raw"].tolist()[:3000]
    labels = df["attack_class"].tolist()[:3000]

    results: dict = {}

    # Track B
    try:
        tok_b  = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
        stats_b = measure_oov(tok_b, texts)
        from collections import defaultdict
        class_texts: dict = defaultdict(list)
        for t, l in zip(texts, labels): class_texts[l].append(t)
        per_b = {c: round(measure_oov(tok_b, ts)["oov_rate"], 5)
                 for c, ts in class_texts.items()}
        results["track_b"] = {**stats_b, "per_class": per_b}
        log.info(f"Track B: oov={stats_b['oov_rate']:.4f}  fertility={stats_b['fertility']:.4f}")
    except Exception as e:
        results["track_b"] = {"error": str(e)}

    # Track A
    track_a_dir = Path(cfg.tokenizer.track_a.output_dir)
    if (track_a_dir / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        tok_a  = AutoTokenizer.from_pretrained(str(track_a_dir))
        stats_a = measure_oov(tok_a, texts)
        results["track_a"] = stats_a
        log.info(f"Track A: oov={stats_a['oov_rate']:.4f}  fertility={stats_a['fertility']:.4f}")

    if "track_a" in results and "track_b" in results:
        for k in ("oov_rate", "fertility", "avg_seq_len"):
            va = results["track_a"].get(k, 0)
            vb = results["track_b"].get(k, 0)
            log.info(f"  {k}: A={va:.4f}  B={vb:.4f}  winner={'B' if vb < va else 'A'}")

    out = Path(cfg.paths.reports) / "metrics" / "tokenizer_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Tokenizer ablation saved to {out}")
    return results



# ─────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────

DISPATCH = {
    "07_tokenizer_ablation":        tokenizer_ablation,
    "08_augmentation_ablation":     augmentation_ablation,
    "09_model_size_scaling":        model_size_scaling,
    "10_label_smoothing_ablation":  label_smoothing_ablation,
}


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    script_name = Path(sys.argv[0]).stem
    fn = DISPATCH.get(script_name)
    if fn is None:
        # Run all
        for name, fn in DISPATCH.items():
            log.info(f"\n{'='*40}\n{name}\n{'='*40}")
            fn(cfg)
    else:
        fn(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())