"""Stage 2.1b — Apply sqlmap-style tamper scripts to seed payloads."""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)

TAMPERS = {
    "apostrophe_mask":  lambda s: s.replace("'", "UTF8MB4_UNICODE_CI"),
    "modsecurity_safe": lambda s: s.replace("=", " LIKE ").replace("OR", "||"),
    "between":          lambda s: s.replace("=1", " BETWEEN 0 AND 2"),
    "ifnull2ifisnull":  lambda s: s.replace("IFNULL(", "IF(ISNULL("),
    "multiplespaces":   lambda s: s.replace(" ", "   "),
    "space2dash":       lambda s: s.replace(" ", "--\n"),
    "space2mssqlblank": lambda s: s.replace(" ", "\\t"),
}

def run(args):
    configure_root(); cfg = load_config(args.config)
    splits_dir = Path(cfg.paths.data_splits)
    if not (splits_dir/"train.parquet").exists(): log.error("No train split"); return
    table = pq.read_table(splits_dir/"train.parquet", filters=[("label","=",1)])
    seeds = [HttpRecord.from_dict(r) for r in table.to_pylist()]
    log.info(f"Loaded {len(seeds):,} seeds")
    records = []
    for name, fn in TAMPERS.items():
        for s in seeds:
            d = s.model_dump(); d["id"] = str(uuid.uuid4())
            d["query_string"] = fn(s.query_string); d["body"] = fn(s.body) if s.body else ""
            d["source"] = f"aug_tamper_{name}"
            records.append(HttpRecord(**d).build_raw())
    out_dir = Path(cfg.paths.data_augmented)/"rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), out_dir/"tamper_scripts.parquet", compression="snappy")
    log.info(f"Generated {len(records):,} tamper-augmented samples")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
