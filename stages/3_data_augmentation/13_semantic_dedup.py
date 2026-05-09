"""Stage 2.6c — Standalone semantic deduplication using MinHash LSH."""
import argparse; from stages.augment.a11_format_validation import run as _run, parse_args as _pa
# Full dedup logic lives in 11_format_validation.py; invoke it.
def parse_args(): return _pa()
if __name__ == "__main__": _run(parse_args())
''',

"stages/3_data_augmentation/14_label_consistency.py": '''"""Stage 2.6d — Standalone label consistency check."""
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq, re
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
_ATTACK = re.compile(r"(union\\s+select|<script|onerror=|\\.\\./etc/passwd|sleep\\s*\\(|exec\\s*\\()", re.I)
def run(args):
    configure_root(); cfg = load_config(args.config)
    p = Path(cfg.paths.data_splits)/"train.parquet"
    if not p.exists(): log.error("No train split"); return
    df = pq.read_table(p, columns=["raw","label","id"]).to_pandas()
    conflicts = df[(df["label"]==0) & df["raw"].str.contains(_ATTACK)]
    log.info(f"Label conflicts (benign with attack pattern): {len(conflicts):,} / {len(df):,}")
    out = Path(cfg.paths.reports)/"metrics"/"label_consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_total": len(df), "n_conflicts": len(conflicts),
                                "conflict_ids": conflicts["id"].tolist()[:20]}, indent=2))
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())

