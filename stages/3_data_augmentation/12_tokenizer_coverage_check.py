"""Stage 2.6b — Standalone tokenizer UNK-rate check (also part of 11_format_validation.py)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    try:
        tok = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    except FileNotFoundError:
        log.error("Track B tokenizer not found. Run Stage 3 first."); return
    aug_dir = Path(cfg.paths.data_augmented)
    all_files = list(aug_dir.rglob("*.parquet"))
    results = {}
    for f in all_files:
        texts = pq.read_table(f, columns=["raw"]).to_pandas()["raw"].tolist()[:1000]
        oov = tok.compute_oov_rate(texts)
        results[f.name] = round(oov, 5)
        status = "OK" if oov <= cfg.augmentation.filtering.max_unk_ratio else "HIGH"
        log.info(f"  {f.name:45s}  OOV={oov:.4f}  [{status}]")
    out = Path(cfg.paths.reports)/"metrics"/"tokenizer_coverage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
''',

"stages/3_data_augmentation/13_semantic_dedup.py": '''"""Stage 2.6c — Standalone semantic deduplication using MinHash LSH."""
import argparse; from stages.augment.a11_format_validation import run as _run, parse_args as _pa
# Full dedup logic lives in 11_format_validation.py; invoke it.
def parse_args(): return _pa()
if __name__ == "__main__": _run(parse_args())
