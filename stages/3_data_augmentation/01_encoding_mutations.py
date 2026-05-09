"""
stages/3_data_augmentation/01_encoding_mutations.py
-----------------------------------------
Stage 2.1 — Rule-based mutation of malicious HTTP requests.

Applies a battery of encoding transformations to each malicious seed:
  - URL encoding (full and partial)
  - Double URL encoding
  - Hex encoding
  - Unicode escape
  - HTML entity encoding
  - SQL comment insertion
  - Case variation
  - Whitespace bypass

Each transformation produces one new sample tagged with
source="aug_rules" and the specific mutation applied.

Run:
    python stages/3_data_augmentation/01_encoding_mutations.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Mutation functions
# ─────────────────────────────────────────────────────────

def mut_url_encode(s: str) -> str:
    """URL-encode all non-alphanumeric characters."""
    return urllib.parse.quote(s, safe="")


def mut_double_url_encode(s: str) -> str:
    return urllib.parse.quote(urllib.parse.quote(s, safe=""), safe="")


def mut_partial_url_encode(s: str, rng: random.Random) -> str:
    """URL-encode only a random 50% of eligible characters."""
    chars = list(s)
    for i, c in enumerate(chars):
        if not c.isalnum() and rng.random() < 0.5:
            chars[i] = urllib.parse.quote(c, safe="")
    return "".join(chars)


def mut_hex_encode(s: str) -> str:
    """Hex-encode alphabetic characters as SQL hex literals (0x61 style)."""
    result = []
    for c in s:
        if c.isalpha():
            result.append(f"0x{ord(c):02x}")
        else:
            result.append(c)
    return "".join(result)


def mut_unicode_escape(s: str) -> str:
    """Unicode-escape alphabetic characters."""
    return "".join(f"\\u{ord(c):04x}" if c.isalpha() else c for c in s)


def mut_html_entity(s: str) -> str:
    """HTML-entity-encode alphabetic chars as decimal entities."""
    return "".join(f"&#{ord(c)};" if c.isalpha() else c for c in s)


def mut_sql_comment_insert(s: str) -> str:
    """Insert /**/ between every whitespace-separated token."""
    return re.sub(r"\s+", "/**/", s)


def mut_case_variation(s: str, rng: random.Random) -> str:
    """Randomly alternate character case."""
    return "".join(
        c.upper() if rng.random() > 0.5 else c.lower()
        for c in s
    )


def mut_whitespace_bypass(s: str, rng: random.Random) -> str:
    """Replace spaces with random whitespace variants."""
    variants = ["\t", "\n", "\r", "  ", " \t", "%09", "%0a", "%0d"]
    return re.sub(r" ", lambda _: rng.choice(variants), s)


def mut_null_byte(s: str) -> str:
    """Inject a null-byte after the first token (path traversal aid)."""
    return s.replace("/", "/%00/", 1)


MUTATION_MAP = {
    "url_encode":         lambda s, rng: mut_url_encode(s),
    "double_url_encode":  lambda s, rng: mut_double_url_encode(s),
    "partial_url_encode": mut_partial_url_encode,
    "hex_encode":         lambda s, rng: mut_hex_encode(s),
    "unicode_escape":     lambda s, rng: mut_unicode_escape(s),
    "html_entity":        lambda s, rng: mut_html_entity(s),
    "comment_insertion":  lambda s, rng: mut_sql_comment_insert(s),
    "case_variation":     mut_case_variation,
    "whitespace_bypass":  mut_whitespace_bypass,
}


# ─────────────────────────────────────────────────────────
# Core augmentation logic
# ─────────────────────────────────────────────────────────

def mutate_record(
    record: HttpRecord,
    mutation_name: str,
    mutation_fn: callable,
    rng: random.Random,
) -> HttpRecord:
    """Apply a mutation to query_string and body; rebuild raw."""
    d = record.model_dump()
    d["id"]           = str(uuid.uuid4())
    d["source"]       = f"aug_rules_{mutation_name}"
    d["query_string"] = mutation_fn(record.query_string, rng)
    if record.body:
        d["body"] = mutation_fn(record.body, rng)

    return HttpRecord(**d).build_raw()


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    rng = random.Random(cfg.project.seed)
    seed_everything(cfg.project.seed)

    aug_cfg = cfg.augmentation.rules

    # Load malicious records from train split
    splits_dir = Path(cfg.paths.data_splits)
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        log.error("train.parquet not found — run data stages first")
        return

    table = pq.read_table(train_path, filters=[("label", "=", 1)])
    seeds = [
        HttpRecord.from_dict(row)
        for row in table.to_pylist()
    ]
    log.info(f"Loaded {len(seeds):,} malicious seed records")

    enabled_mutations = {
        k: v for k, v in MUTATION_MAP.items()
        if k in aug_cfg.encodings
    }
    log.info(f"Applying {len(enabled_mutations)} mutation types: {list(enabled_mutations.keys())}")

    out_dir = Path(cfg.paths.data_augmented) / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_augmented: list[HttpRecord] = []
    per_mutation_counts: dict[str, int] = {}

    for mut_name, mut_fn in enabled_mutations.items():
        mutated = []
        for seed_record in seeds:
            try:
                new_record = mutate_record(seed_record, mut_name, mut_fn, rng)
                mutated.append(new_record)
            except Exception as e:
                log.debug(f"Mutation '{mut_name}' failed on record {seed_record.id}: {e}")

        per_mutation_counts[mut_name] = len(mutated)
        all_augmented.extend(mutated)
        log.info(f"  {mut_name:25s}: {len(mutated):,} new samples")

    # Write output
    if all_augmented:
        table = records_to_table(all_augmented)
        out_path = out_dir / "rule_mutations.parquet"
        pq.write_table(table, out_path, compression="snappy")
        log.info(f"Total: {len(all_augmented):,} augmented records → {out_path}")

    # Stats
    stats_path = Path(cfg.paths.reports) / "metrics" / "augmentation_rules.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "n_seeds":       len(seeds),
        "n_augmented":   len(all_augmented),
        "per_mutation":  per_mutation_counts,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())