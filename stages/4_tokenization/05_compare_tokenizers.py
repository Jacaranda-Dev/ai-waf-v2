"""Stage 3.4 — Side-by-side tokenizer comparison report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.tokenizer.vocab_utils import compare_tokenizers
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    track_a_dir = Path(cfg.tokenizer.track_a.output_dir)
    track_b_dir = Path(cfg.tokenizer.track_b.output_dir)
    if not (track_a_dir/"tokenizer_config.json").exists():
        log.error("Track A tokenizer missing"); return
    if not (track_b_dir/"tokenizer.json").exists():
        log.error("Track B tokenizer missing"); return
    from transformers import AutoTokenizer
    tok_a = AutoTokenizer.from_pretrained(str(track_a_dir))
    tok_b = HttpTokenizer.load(str(track_b_dir), cfg.tokenizer.seq_len)
    val_path = Path(cfg.paths.data_splits)/"val.parquet"
    df = pq.read_table(val_path, columns=["raw","attack_class"]).to_pandas()
    texts, labels = df["raw"].tolist()[:3000], df["attack_class"].tolist()[:3000]
    result = compare_tokenizers(tok_a, tok_b, texts, labels)
    log.info(f"OOV winner:      {result.get(\'winner_oov\')}")
    log.info(f"Fertility winner:{result.get(\'winner_fertility\')}")
    if "vocab_jaccard_overlap" in result:
        log.info(f"Vocab overlap:   {result[\'vocab_jaccard_overlap\']:.4f}")
    out = Path(cfg.paths.reports)/"metrics"/"tokenizer_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())