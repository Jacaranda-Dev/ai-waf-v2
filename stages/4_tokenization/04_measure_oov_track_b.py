"""Stage 3.4 — Measure OOV rate and fertility for Track B tokenizer."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.tokenizer.vocab_utils import measure_oov
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    try:
        tok = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    except FileNotFoundError:
        log.error("Track B tokenizer not found. Run 03_train_custom_bpe.py first."); return
    val_path = Path(cfg.paths.data_splits)/"val.parquet"
    if not val_path.exists(): log.error("No val split"); return
    df = pq.read_table(val_path, columns=["raw","attack_class"]).to_pandas()
    texts  = df["raw"].tolist()[:5000]
    labels = df["attack_class"].tolist()[:5000]
    stats  = measure_oov(tok, texts)
    log.info(f"Track B: oov={stats[\'oov_rate\']:.4f}  fertility={stats[\'fertility\']:.4f}")
    class_texts = defaultdict(list)
    for t, l in zip(texts, labels): class_texts[l].append(t)
    per_class = {cls: round(measure_oov(tok,ts)["oov_rate"],6) for cls,ts in class_texts.items()}
    result = {**stats, "per_class_oov": per_class, "vocab_size": tok.vocab_size}
    out = Path(cfg.paths.reports)/"metrics"/"tokenizer_oov_track_b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())