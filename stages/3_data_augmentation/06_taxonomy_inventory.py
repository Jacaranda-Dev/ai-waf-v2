"""
stages/3_data_augmentation/16_taxonomy_inventory.py
----------------------------------------------------
Enhanced Taxonomy Inventory + Distribution Shift Report

Improvements over the original:
  1. Jensen-Shannon Divergence (JSD) between original and augmented class distributions
     → quantifies how much augmentation has shifted the data manifold
  2. Per-source breakdown showing which augmentation pipeline contributed what
  3. Imbalance metrics (Gini coefficient of class counts)
  4. Outputs augmentation targets for the Governor (01_attack_synthesis.py)
  5. Eliminated Stage 2 duplicate: this version is the single authoritative inventory

Run:
    python stages/3_data_augmentation/16_taxonomy_inventory.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

MIN_SAMPLES_PER_CLASS = 500


# ─────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon Divergence
# ─────────────────────────────────────────────────────────────────────────────

def _kl_div(p: dict[str, float], q: dict[str, float]) -> float:
    """KL divergence D(P||Q); smoothed to avoid log(0)."""
    eps = 1e-10
    keys = set(p) | set(q)
    return sum(
        p.get(k, eps) * math.log((p.get(k, eps) + eps) / (q.get(k, eps) + eps))
        for k in keys
    )


def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """
    JSD(P || Q) ∈ [0, 1]  (using log base 2, so result is in bits Normalized to 1).
    JSD = 0 → distributions identical.
    JSD = 1 → distributions completely disjoint.
    """
    keys = set(p) | set(q)
    total_p = sum(p.values()) or 1.0
    total_q = sum(q.values()) or 1.0
    p_norm = {k: p.get(k, 0) / total_p for k in keys}
    q_norm = {k: q.get(k, 0) / total_q for k in keys}
    m = {k: (p_norm[k] + q_norm[k]) / 2.0 for k in keys}
    jsd = 0.5 * _kl_div(p_norm, m) + 0.5 * _kl_div(q_norm, m)
    # Normalize to [0, 1] (log2 basis → divide by log(2) since we used natural log)
    return round(min(1.0, jsd / math.log(2)), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Gini coefficient  (class imbalance measure)
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(counts: dict) -> float:
    """
    Gini coefficient of class sample counts.
    0 = perfectly balanced; 1 = all samples in one class.
    """
    values = sorted(counts.values())
    n      = len(values)
    if n == 0:
        return 0.0
    total  = sum(values) or 1
    cumsum = 0
    for i, v in enumerate(values, 1):
        cumsum += v * (2 * i - n - 1)
    return round(abs(cumsum) / (n * total), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    # ── Load base (pre-augmentation) data ─────────────────────────────────
    base_paths = [
        Path(cfg.paths.data_normalized) / "deduped.parquet",
        Path(cfg.paths.data_normalized) / "all_datasets.parquet",
    ]
    base_path = next((p for p in base_paths if p.exists()), None)
    if not base_path:
        log.error("No Normalized base data found — run Stage 1 first")
        return

    base_table   = pq.read_table(base_path, columns=["label", "attack_class", "source"])
    base_classes = base_table["attack_class"].to_pylist()
    base_labels  = base_table["label"].to_pylist()
    base_sources = base_table["source"].to_pylist()

    base_class_counts = Counter(c for c in base_classes if c != "benign")
    base_benign       = sum(1 for l in base_labels if l == 0)
    base_malicious    = sum(1 for l in base_labels if l == 1)

    # ── Load augmented data ────────────────────────────────────────────────
    aug_dir = Path(cfg.paths.data_augmented)
    aug_files = list(aug_dir.rglob("*.parquet"))

    aug_class_counts: Counter = Counter()
    aug_source_counts: Counter = Counter()

    for fp in aug_files:
        try:
            t = pq.read_table(fp, columns=["attack_class", "source"])
            aug_class_counts.update(
                c for c in t["attack_class"].to_pylist() if c and c != "benign"
            )
            aug_source_counts.update(t["source"].to_pylist())
        except Exception as e:
            log.warning(f"Could not read {fp}: {e}")

    # ── Combined ───────────────────────────────────────────────────────────
    combined_class_counts: Counter = Counter()
    combined_class_counts.update(base_class_counts)
    combined_class_counts.update(aug_class_counts)

    # ── Required classes from config ───────────────────────────────────────
    required  = set(cfg.data.data_schema.attack_classes)  
    present   = {cls for cls, cnt in combined_class_counts.items() if cnt > 0}
    missing   = required - present
    low_count = {
        cls: int(cnt) for cls, cnt in combined_class_counts.items()
        if 0 < cnt < MIN_SAMPLES_PER_CLASS
    }

    # ── Jensen-Shannon Divergence: base vs. combined ───────────────────────
    jsd = jensen_shannon_divergence(
        dict(base_class_counts),
        dict(combined_class_counts),
    )
    jsd_interpretation = (
        "minimal shift (< 0.05)"  if jsd < 0.05 else
        "moderate shift (< 0.15)" if jsd < 0.15 else
        "large shift — review augmentation strategy"
    )

    # ── Gini ───────────────────────────────────────────────────────────────
    gini_base     = gini_coefficient(dict(base_class_counts))
    gini_combined = gini_coefficient(dict(combined_class_counts))

    # ── Augmentation target output for Governor ────────────────────────────
    target_per_class = getattr(getattr(cfg.augmentation, None, None), "target_per_class", 5_000) if hasattr(cfg, "augmentation") else 5_000
    augmentation_targets = {
        cls: max(0, target_per_class - combined_class_counts.get(cls, 0))
        for cls in required
    }

    # ── Console report ─────────────────────────────────────────────────────
    log.info("\n=== TAXONOMY INVENTORY + DISTRIBUTION SHIFT REPORT ===")
    log.info(f"{'Class':25s}  {'Base':>8}  {'Aug':>8}  {'Combined':>10}  {'Status':>10}")
    log.info("-" * 70)
    for cls in sorted(required | present):
        base_n   = base_class_counts.get(cls, 0)
        aug_n    = aug_class_counts.get(cls, 0)
        total_n  = combined_class_counts.get(cls, 0)
        status   = ("OK"      if total_n >= MIN_SAMPLES_PER_CLASS
                    else "LOW"  if total_n > 0
                    else "MISSING")
        log.info(f"{cls:25s}  {base_n:>8,}  {aug_n:>8,}  {total_n:>10,}  {status:>10}")

    log.info(f"\nBenign (base):    {base_benign:,}")
    log.info(f"Malicious (base): {base_malicious:,}")
    log.info(f"\nJensen-Shannon Divergence (base vs combined): {jsd:.6f}  →  {jsd_interpretation}")
    log.info(f"Gini (base): {gini_base:.4f}  →  Gini (combined): {gini_combined:.4f}")
    log.info(f"Gini delta: {gini_combined - gini_base:+.4f}  ({'more balanced' if gini_combined < gini_base else 'less balanced'})")
    log.info(f"\nMissing classes:   {sorted(missing)}")
    log.info(f"Low-count classes: {low_count}")

    result = {
        "class_counts": {
            "base":     dict(base_class_counts),
            "augmented": dict(aug_class_counts),
            "combined": dict(combined_class_counts),
        },
        "source_counts":       dict(aug_source_counts),
        "required_classes":    sorted(required),
        "present_classes":     sorted(present),
        "missing_classes":     sorted(missing),
        "low_count_classes":   low_count,
        "distribution_shift": {
            "jensen_shannon_divergence": jsd,
            "interpretation":            jsd_interpretation,
            "gini_base":                 gini_base,
            "gini_combined":             gini_combined,
            "gini_delta":                round(gini_combined - gini_base, 6),
        },
        "augmentation_targets": augmentation_targets,
        "totals": {
            "base_benign":    base_benign,
            "base_malicious": base_malicious,
            "base_total":     len(base_labels),
            "aug_total":      sum(aug_class_counts.values()),
        },
    }

    out = Path(cfg.paths.reports) / "metrics" / "taxonomy_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"\nInventory + distribution shift report saved → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())