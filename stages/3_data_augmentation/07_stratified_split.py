"""
stages/3_data_augmentation/17_stratified_split.py
--------------------------------------------------
Enhanced Stratified Train / Val / Test / Adversarial / Canary Split

Improvements over original:
  1. Composite stratification key: label × attack_class × source_bucket
     → prevents any single dataset (e.g., CSIC 2010) from dominating one split
     → source_bucket coarsens source strings to a manageable stratum count
  2. MinHash LSH proximity filter: enforces Jaccard(train, test) < 0.70
     on a per-record basis after the split, eliminating data leakage
  3. Split health report: per-split breakdown by attack_class × source_bucket,
     plus class imbalance ratio check (warns if any split > 2× the global ratio)

Run:
    python stages/3_data_augmentation/17_stratified_split.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

LEAKAGE_THRESHOLD = 0.70   # Jaccard similarity above this → remove from test/canary


# ─────────────────────────────────────────────────────────────────────────────
# Source bucket mapper
# ─────────────────────────────────────────────────────────────────────────────

def _source_bucket(source: str) -> str:
    """
    Coarsen raw source strings into a small set of buckets so the composite
    stratification key stays manageable.  Without this, hundreds of unique
    aug_* source strings create single-member strata that crash train_test_split.
    """
    s = source.lower()
    if s.startswith("aug_synthesis"):  return "aug_synthesis"
    if s.startswith("aug_benign"):     return "aug_benign"
    if s.startswith("aug_grammar"):    return "aug_grammar"
    if s.startswith("aug_tamper"):     return "aug_tamper"
    if s.startswith("aug_api_llm"):    return "aug_llm"
    if s.startswith("aug_local_llm"):  return "aug_llm"
    if s.startswith("aug_rules"):      return "aug_rules"
    # Known real datasets
    if "csic"   in s: return "real_csic"
    if "ecml"   in s: return "real_ecml"
    if "wamm"   in s: return "real_wamm"
    if "kaggle" in s: return "real_kaggle"
    return "real_other"


def _add_composite_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_src_bucket"]  = df["source"].apply(_source_bucket)
    df["_strat_key"]   = (
        df["label"].astype(str) + "_"
        + df["attack_class"].astype(str) + "_"
        + df["_src_bucket"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MinHash utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(max(1, len(text) - 2)):
        m.update(text[i:i+3].encode("utf-8", errors="replace"))
    return m


def _build_train_lsh(train_df: pd.DataFrame, threshold: float, num_perm: int = 64) -> MinHashLSH:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for _, row in train_df.iterrows():
        mh = _build_minhash(row["raw"] or "", num_perm)
        try:
            lsh.insert(row["id"], mh)
        except Exception:
            pass
    return lsh


def _filter_leakage(
    df:         pd.DataFrame,
    train_lsh:  MinHashLSH,
    split_name: str,
    num_perm:   int = 64,
) -> pd.DataFrame:
    """Remove rows from df that are near-duplicates of training records."""
    clean_ids = []
    n_removed = 0
    for _, row in df.iterrows():
        mh = _build_minhash(row["raw"] or "", num_perm)
        if train_lsh.query(mh):
            n_removed += 1
        else:
            clean_ids.append(row["id"])

    if n_removed:
        log.warning(
            f"Leakage filter removed {n_removed:,} records from "
            f"{split_name} (Jaccard > {LEAKAGE_THRESHOLD:.2f} with train)"
        )
    return df[df["id"].isin(clean_ids)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split helper
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_split(
    df:        pd.DataFrame,
    test_size: float,
    seed:      int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on _strat_key.
    Falls back gracefully when a stratum has < 2 members.
    """
    strat = df["_strat_key"]
    # Drop strata that can't be split (< 2 members)
    counts     = strat.value_counts()
    valid_strat = strat.where(strat.map(counts) >= 2)

    try:
        main, held = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=valid_strat,
        )
    except ValueError as e:
        log.warning(f"Composite stratification failed ({e}) — falling back to label+class stratification")
        simple_strat = (df["label"].astype(str) + "_" + df["attack_class"].astype(str))
        try:
            main, held = train_test_split(
                df, test_size=test_size, random_state=seed,
                stratify=simple_strat,
            )
        except ValueError:
            log.warning("Simple stratification also failed — using random split")
            main, held = train_test_split(df, test_size=test_size, random_state=seed)

    return main.copy(), held.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Split health report
# ─────────────────────────────────────────────────────────────────────────────

def _split_health(
    splits:      dict[str, pd.DataFrame],
    global_ratio: float,
) -> dict:
    """
    For each split, compute:
      - n_total, n_benign, n_malicious, imbalance_ratio
      - class × source_bucket breakdown
      - warn if imbalance_ratio > 2 × global_ratio
    """
    report = {}
    for name, df in splits.items():
        n_ben  = int((df["label"] == 0).sum())
        n_mal  = int((df["label"] == 1).sum())
        ratio  = round(n_ben / max(1, n_mal), 4)
        skewed = ratio > 2.0 * global_ratio or ratio < 0.5 * global_ratio

        class_source = (
            df.groupby(["attack_class", "_src_bucket"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        report[name] = {
            "n_total":        int(len(df)),
            "n_benign":       n_ben,
            "n_malicious":    n_mal,
            "imbalance_ratio": ratio,
            "imbalance_warning": skewed,
            "class_source_breakdown": class_source,
        }
        if skewed:
            log.warning(
                f"Split '{name}' imbalance ratio {ratio:.2f} deviates from "
                f"global {global_ratio:.2f} by >2×"
            )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg  = load_config(args.config)
    seed = cfg.project.seed
    seed_everything(seed)

    require_inputs({
        "data/filtered/filtered.parquet": "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.data_splits) / "train.parquet",
        args.force, "Stage 3.7 stratified split"
    ):
        return

    # Input: filtered data preferred; fall back to deduped
    filtered_path = Path(cfg.paths.data_filtered) / "filtered.parquet"
    deduped_path  = Path(cfg.paths.data_normalized) / "deduped.parquet"
    in_path = filtered_path if filtered_path.exists() else deduped_path

    if not in_path.exists():
        log.error(f"No input data at {in_path} — run the quality gate first")
        return

    splits_dir = Path(cfg.paths.data_splits)
    splits_dir.mkdir(parents=True, exist_ok=True)

    timer = StepTimer()

    with timer.step("load_parquet"):
        df = pq.read_table(in_path).to_pandas()
    log.info(f"Loaded {len(df):,} records from {in_path.name}")

    df = _add_composite_key(df)

    scfg = cfg.data.split

    # ── Sequential splits ─────────────────────────────────────────────────
    with timer.step("stratified_split"):
        # Step 1: canary (2%)
        df_main, df_canary = _stratified_split(df, test_size=scfg.canary, seed=seed)

        # Step 2: adversarial (3% of remaining)
        adv_frac = scfg.adversarial / (scfg.train + scfg.val + scfg.test + scfg.adversarial)
        df_main, df_adversarial = _stratified_split(df_main, test_size=adv_frac, seed=seed)

        # Step 3: test (15% of remaining)
        test_frac = scfg.test / (scfg.train + scfg.val + scfg.test)
        df_main, df_test = _stratified_split(df_main, test_size=test_frac, seed=seed)

        # Step 4: val (10% of remaining)
        val_frac = scfg.val / (scfg.train + scfg.val)
        df_train, df_val = _stratified_split(df_main, test_size=val_frac, seed=seed)

    # ── Build train MinHash LSH for leakage filter ─────────────────────────
    log.info(f"Building train LSH index ({len(df_train):,} records) for leakage guard...")
    with timer.step("build_leakage_lsh"):
        train_lsh = _build_train_lsh(df_train, threshold=LEAKAGE_THRESHOLD)

    # ── Apply leakage filter to test and canary ────────────────────────────
    with timer.step("leakage_filter"):
        df_test   = _filter_leakage(df_test,   train_lsh, "test")
        df_canary = _filter_leakage(df_canary, train_lsh, "canary")

    # ── Write splits ──────────────────────────────────────────────────────
    _DROP_COLS = ["_strat_key", "_src_bucket"]

    split_dfs: dict[str, pd.DataFrame] = {
        "train":       df_train,
        "val":         df_val,
        "test":        df_test,
        "adversarial": df_adversarial,
        "canary":      df_canary,
    }

    with timer.step("parquet_write"):
        for name, sdf in split_dfs.items():
            sdf = sdf.drop(columns=_DROP_COLS, errors="ignore").copy()
            sdf["split"] = name
            pq.write_table(
                pa.Table.from_pandas(sdf, preserve_index=False),
                splits_dir / f"{name}.parquet",
                compression="snappy",
            )
            n_ben = int((sdf["label"] == 0).sum())
            n_mal = int((sdf["label"] == 1).sum())
            log.info(f"  {name:12s}: {len(sdf):>7,}  benign={n_ben:,}  malicious={n_mal:,}")

    # ── Health report ─────────────────────────────────────────────────────
    global_ben  = int((df["label"] == 0).sum())
    global_mal  = int((df["label"] == 1).sum())
    global_ratio = round(global_ben / max(1, global_mal), 4)

    # Re-add composite key columns for health report (they were dropped for output)
    health_dfs = {}
    for name, sdf in split_dfs.items():
        tmp = sdf.copy()
        tmp["_src_bucket"] = tmp["source"].apply(_source_bucket)
        health_dfs[name] = tmp

    health = _split_health(health_dfs, global_ratio)
    health["global_imbalance_ratio"] = global_ratio
    health["leakage_threshold"]      = LEAKAGE_THRESHOLD
    health["stratification_key"]     = "label × attack_class × source_bucket"

    stats_path = Path(cfg.paths.reports) / "metrics" / "split_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    health["timings_s"] = timer.timings
    stats_path.write_text(json.dumps(health, indent=2))
    log.info(f"Split health report → {stats_path}")

    warnings = [name for name, info in health.items()
                if isinstance(info, dict) and info.get("imbalance_warning")]
    if warnings:
        log.warning(f"Imbalance warnings for splits: {warnings}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="07_stratified_split"):
            mlflow.log_params({
                "leakage_threshold": LEAKAGE_THRESHOLD,
                "stratification_key": "label × attack_class × source_bucket",
                "global_imbalance_ratio": global_ratio,
                "n_imbalance_warnings": len(warnings),
            })
            metrics: dict[str, float] = {"global_imbalance_ratio": float(global_ratio)}
            for split_name, info in health.items():
                if isinstance(info, dict) and "n_total" in info:
                    metrics[f"{split_name}_n_total"]   = float(info["n_total"])
                    metrics[f"{split_name}_n_benign"]  = float(info["n_benign"])
                    metrics[f"{split_name}_n_malicious"] = float(info["n_malicious"])
                    metrics[f"{split_name}_imbalance"] = float(info["imbalance_ratio"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(stats_path))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())