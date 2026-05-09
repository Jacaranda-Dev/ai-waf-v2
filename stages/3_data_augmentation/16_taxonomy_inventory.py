"""
stages/2_baselines/06_taxonomy_inventory.py
-------------------------------------------
Stage 0.4 — Enumerate which attack classes are present in the
collected dataset and measure coverage gaps.

Output: reports/metrics/taxonomy_inventory.json
        Reports missing classes → drives augmentation targeting.

Run:
    python stages/2_baselines/06_taxonomy_inventory.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

MIN_SAMPLES_PER_CLASS = 500   # below this → needs augmentation

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    norm_path = Path(cfg.paths.data_normalized) / "deduped.parquet"
    if not norm_path.exists():
        norm_path = Path(cfg.paths.data_normalized) / "all_datasets.parquet"
    if not norm_path.exists():
        log.error("No normalized data found — run Stage 1 first")
        return

    table   = pq.read_table(norm_path, columns=["label", "attack_class", "source"])
    labels  = table["label"].to_pylist()
    classes = table["attack_class"].to_pylist()
    sources = table["source"].to_pylist()

    class_counts  = Counter(classes)
    source_counts = Counter(sources)

    required  = set(cfg.data.schema.attack_classes)
    present   = {cls for cls, cnt in class_counts.items() if cls != "benign" and cnt > 0}
    missing   = required - present
    low_count = {cls: cnt for cls, cnt in class_counts.items()
                 if cls != "benign" and 0 < cnt < MIN_SAMPLES_PER_CLASS}

    log.info("\n=== TAXONOMY INVENTORY ===")
    log.info(f"{'Class':25s}  {'Count':>8}  {'Status':>12}")
    log.info("-" * 50)
    for cls in sorted(required | present):
        cnt    = class_counts.get(cls, 0)
        status = "OK" if cnt >= MIN_SAMPLES_PER_CLASS else ("LOW" if cnt > 0 else "MISSING")
        log.info(f"{cls:25s}  {cnt:>8,}  {status:>12}")

    log.info(f"\nBenign samples: {class_counts.get('benign', 0):,}")
    log.info(f"\nMissing classes (need full augmentation): {sorted(missing)}")
    log.info(f"Low-count classes (need more augmentation): {dict(sorted(low_count.items()))}")

    result = {
        "class_counts":   dict(class_counts),
        "source_counts":  dict(source_counts),
        "required_classes": sorted(required),
        "present_classes":  sorted(present),
        "missing_classes":  sorted(missing),
        "low_count_classes": low_count,
        "total_samples":    len(labels),
        "benign":           labels.count(0),
        "malicious":        labels.count(1),
    }

    out = Path(cfg.paths.reports) / "metrics" / "taxonomy_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"\nTaxonomy inventory saved to {out}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())