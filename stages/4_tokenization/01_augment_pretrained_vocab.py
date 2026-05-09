"""Stage 3.1 — Augment BERT-base vocab with HTTP-specific tokens (Track A)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ai_waf_v2.tokenizer.vocab_utils import augment_pretrained_vocab
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    tok_a = cfg.tokenizer.track_a
    tokenizer, n_added = augment_pretrained_vocab(
        base_model=tok_a.base_model,
        new_tokens=tok_a.http_tokens,
        output_dir=tok_a.output_dir,
    )
    log.info(f"Track A: {n_added} new tokens added to {tok_a.base_model} vocab")
    log.info(f"New vocab size: {len(tokenizer)}")
    stats = {"base_model": tok_a.base_model, "n_added": n_added,
             "new_vocab_size": len(tokenizer), "tokens_added": tok_a.http_tokens}
    out = Path(cfg.paths.reports)/"metrics"/"tokenizer_track_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())