"""
stages/2_baselines/00_stratified_split.py
------------------------------------
Stage 2.0 — Stratified train / val / test / adversarial / canary split.

Stratifies by (label × attack_class) so every split has proportional
representation of each attack type and class balance.

Writes five Parquet files to data/splits/:
    train.parquet, val.parquet, test.parquet,
    adversarial.parquet, canary.parquet

Run:
    python stages/1_data_acquisition_and_curation/04_stratified_split.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    # Try filtered data first; fall back to deduped
    filtered_path = Path(cfg.paths.data_filtered) / "filtered.parquet"
    deduped_path  = Path(cfg.paths.data_normalized) / "deduped.parquet"

    in_path = filtered_path if filtered_path.exists() else deduped_path
    if not in_path.exists():
        log.error(f"No input found at {in_path} — run earlier stages first")
        return

    splits_dir = Path(cfg.paths.data_splits)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(in_path).to_pandas()
    log.info(f"Loaded {len(df):,} records from {in_path.name}")

    # Stratification key: label + attack_class
    df["_strat_key"] = df["label"].astype(str) + "_" + df["attack_class"].astype(str)

    scfg        = cfg.data.split
    random_seed = cfg.project.seed

    # Sequential splits using sklearn train_test_split
    # Step 1: carve out canary (2%) from the full set
    df_main, df_canary, fb_canary = _stratified_split(
        df, test_size=scfg.canary, seed=random_seed
    )

    # Step 2: carve out adversarial holdout (3%) from remaining
    remaining_total = scfg.train + scfg.val + scfg.test + scfg.adversarial
    adv_frac_of_remaining = scfg.adversarial / remaining_total
    df_main, df_adversarial, fb_adversarial = _stratified_split(
        df_main, test_size=adv_frac_of_remaining, seed=random_seed
    )

    # Step 3: test split
    remaining_total2 = scfg.train + scfg.val + scfg.test
    test_frac = scfg.test / remaining_total2
    df_main, df_test, fb_test  = _stratified_split(
        df_main, test_size=test_frac, seed=random_seed
    )

    # Step 4: val split
    remaining_total3 = scfg.train + scfg.val
    val_frac = scfg.val / remaining_total3
    df_train, df_val, fb_val = _stratified_split(
        df_main, test_size=val_frac, seed=random_seed
    )

    splits = {
        "train":      df_train,
        "val":        df_val,
        "test":       df_test,
        "adversarial": df_adversarial,
        "canary":     df_canary,
    }

      # Collect which splits fell back — canary is carved first and most
    # likely to trigger it if a rare class has only 1 sample.
    fallback_splits = [
        name for name, triggered in [
            ("canary", fb_canary),
            ("adversarial", fb_adversarial),
            ("test", fb_test),
            ("val", fb_val),
        ] if triggered
    ]
    if fallback_splits:
        log.warning(f"Random fallback used for splits: {fallback_splits}")


    split_stats: dict[str, dict] = {}

    

    for split_name, split_df in splits.items():
        # Tag the split column
        split_df = split_df.drop(columns=["_strat_key"], errors="ignore").copy()
        split_df["split"] = split_name

        out_path = splits_dir / f"{split_name}.parquet"
        pq.write_table(
            pa.Table.from_pandas(split_df, preserve_index=False),
            out_path,
            compression="snappy",
        )

        n_total    = len(split_df)
        n_benign   = (split_df["label"] == 0).sum()
        n_malicious = (split_df["label"] == 1).sum()

        split_stats[split_name] = {
            "n_total":      int(n_total),
            "n_benign":     int(n_benign),
            "n_malicious":  int(n_malicious),
            "imbalance_ratio": round(n_benign / max(1, n_malicious), 2),
            "attack_class_counts": split_df["attack_class"].value_counts().to_dict(),
        }
        log.info(
            f"  {split_name:12s}: {n_total:>7,} "
            f"(benign={n_benign:,}, malicious={n_malicious:,})"
        )

    stats_meta = {
        "fallback_splits": fallback_splits,          # [] means fully stratified
        "stratification_key": "label × attack_class",
    }
    # Save stats
    stats_path = Path(cfg.paths.reports) / "metrics" / "split_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({**split_stats, "_meta": stats_meta}, indent=2))
    log.info(f"Split stats saved to {stats_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="00_stratified_split"):
            mlflow.log_params({
                "split_train":      scfg.train,
                "split_val":        scfg.val,
                "split_test":       scfg.test,
                "split_adversarial": scfg.adversarial,
                "split_canary":     scfg.canary,
                "fallback_splits":  fallback_splits,
            })
            metrics: dict[str, float] = {}
            for split_name, info in split_stats.items():
                metrics[f"{split_name}_n_total"]    = float(info["n_total"])
                metrics[f"{split_name}_n_benign"]   = float(info["n_benign"])
                metrics[f"{split_name}_n_malicious"] = float(info["n_malicious"])
                metrics[f"{split_name}_imbalance"]  = float(info["imbalance_ratio"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(stats_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def _stratified_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Returns (main, held, used_fallback).
    used_fallback is True when stratification failed and a plain random
    split was used instead — callers should record this for auditing.
    """
    strat = df["_strat_key"]
    # Drop stratification keys that have only 1 sample (can't be split)
    valid_strat = strat.where(strat.map(strat.value_counts()) >= 2)

    try:
        main, held = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=valid_strat,
        )
        return main.copy(), held.copy(), False  # ← no fallback

    except ValueError:
        log.warning("Stratified split failed — falling back to random split")
        main, held = train_test_split(df, test_size=test_size, random_state=seed)
        return main.copy(), held.copy(), True    # ← fallback triggered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())