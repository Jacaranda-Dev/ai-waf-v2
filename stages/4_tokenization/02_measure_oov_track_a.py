"""Stage 3.2 — Measure OOV rate and fertility for Track A tokenizer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.tokenizer.vocab_utils import measure_oov
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    track_a_dir = Path(cfg.tokenizer.track_a.output_dir)
    if not (track_a_dir/"tokenizer_config.json").exists():
        log.error("Track A tokenizer not found. Run 01_augment_pretrained_vocab.py first."); return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(track_a_dir))
    val_path  = Path(cfg.paths.data_splits)/"val.parquet"
    if not val_path.exists(): log.error("No val split"); return
    df = pq.read_table(val_path, columns=["raw","attack_class"]).to_pandas()
    texts  = df["raw"].tolist()[:5000]
    labels = df["attack_class"].tolist()[:5000]
    stats  = measure_oov(tokenizer, texts)
    log.info(f"Track A OOV: {stats}")
    # Per-class
    from collections import defaultdict
    class_texts = defaultdict(list)
    for t, l in zip(texts, labels): class_texts[l].append(t)
    per_class = {}
    for cls, ts in class_texts.items():
        per_class[cls] = round(measure_oov(tokenizer, ts)["oov_rate"], 6)
    result = {**stats, "per_class_oov": per_class}
    out = Path(cfg.paths.reports)/"metrics"/"tokenizer_oov_track_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Track A OOV results saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())