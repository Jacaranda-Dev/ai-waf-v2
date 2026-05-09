"""

stages/7_evaluation/08_augmentation_ablation.py
-----------------------------------------
All four ablation studies in one file (dispatched by script name).
Each ablation trains a fast probe model (LR + TF-IDF) to isolate
the variable of interest without requiring full transformer training.

For full transformer ablations, use MLflow to compare existing runs:
these scripts compare saved checkpoints and report metrics.

Run:
    python stages/7_evaluation/08_augmentation_ablation.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)



# ─────────────────────────────────────────────────────────
# 08 — Augmentation ablation
# ─────────────────────────────────────────────────────────

def augmentation_ablation(cfg) -> dict:
    """
    Train a fast TF-IDF + LR probe for each augmentation source disabled.
    Measures how much each augmentation method contributes to AUC-PR.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from ai_waf_v2.eval.metrics import compute_metrics

    splits_dir = Path(cfg.paths.data_splits)
    if not (splits_dir / "train.parquet").exists():
        return {"error": "train.parquet not found"}

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw","label","source"])
    val_df   = pd.read_parquet(splits_dir / "val.parquet",   columns=["raw","label"])
    X_val, y_val = val_df["raw"].tolist(), val_df["label"].to_numpy()

    def probe(X_train, y_train):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3), max_features=20_000)),
            ("clf",   LogisticRegression(max_iter=300, C=1.0, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return compute_metrics(torch.tensor(preds), torch.tensor(proba, dtype=torch.float),
                               torch.tensor(y_val))

    aug_sources = [s for s in train_df["source"].unique() if s.startswith("aug_")]
    results: dict = {}

    # Full training set
    m_full = probe(train_df["raw"].tolist(), train_df["label"].to_numpy())
    results["full"] = {"auc_pr": m_full["auc_pr"], "f1": m_full["f1"]}
    log.info(f"  {'full':35s}: AUC-PR={m_full['auc_pr']:.4f}")

    # Drop each augmentation source one at a time
    for source in aug_sources:
        subset = train_df[train_df["source"] != source]
        if len(subset) < 100: continue
        m = probe(subset["raw"].tolist(), subset["label"].to_numpy())
        delta = m["auc_pr"] - m_full["auc_pr"]
        results[f"drop_{source}"] = {"auc_pr": m["auc_pr"], "delta": round(delta,5)}
        log.info(f"  {'drop_'+source:35s}: AUC-PR={m['auc_pr']:.4f}  delta={delta:+.4f}")

    out = Path(cfg.paths.reports) / "metrics" / "augmentation_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Augmentation ablation saved to {out}")
    return results




# ─────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────

DISPATCH = {
    "07_tokenizer_ablation":        tokenizer_ablation,
    "08_augmentation_ablation":     augmentation_ablation,
    "09_model_size_scaling":        model_size_scaling,
    "10_label_smoothing_ablation":  label_smoothing_ablation,
}


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    script_name = Path(sys.argv[0]).stem
    fn = DISPATCH.get(script_name)
    if fn is None:
        # Run all
        for name, fn in DISPATCH.items():
            log.info(f"\n{'='*40}\n{name}\n{'='*40}")
            fn(cfg)
    else:
        fn(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())