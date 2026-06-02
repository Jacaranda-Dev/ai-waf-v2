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
import argparse, json
from pathlib import Path
import torch

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

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

    timer = StepTimer()
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
    with timer.step("probe_full"):
        m_full = probe(train_df["raw"].tolist(), train_df["label"].to_numpy())
    results["full"] = {"auc_pr": m_full["auc_pr"], "f1": m_full["f1"]}
    log.info(f"  {'full':35s}: AUC-PR={m_full['auc_pr']:.4f}")

    # Drop each augmentation source one at a time
    with timer.step("probe_ablations"):
        for source in aug_sources:
            subset = train_df[train_df["source"] != source]
            if len(subset) < 100: continue
            m = probe(subset["raw"].tolist(), subset["label"].to_numpy())
            delta = m["auc_pr"] - m_full["auc_pr"]
            results[f"drop_{source}"] = {"auc_pr": m["auc_pr"], "delta": round(delta,5)}
            log.info(f"  {'drop_'+source:35s}: AUC-PR={m['auc_pr']:.4f}  delta={delta:+.4f}")

    results["timings_s"] = timer.timings
    out = Path(cfg.paths.reports) / "metrics" / "augmentation_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Augmentation ablation saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="08_augmentation_ablation") as _run:
            mlflow.log_params({
                "n_aug_sources": len(aug_sources),
            })
            metrics: dict[str, float] = {}
            full_m = results.get("full", {})
            if full_m:
                metrics["full_auc_pr"] = float(full_m.get("auc_pr", 0))
                metrics["full_f1"]     = float(full_m.get("f1", 0))
            for key, val in results.items():
                if key.startswith("drop_") and isinstance(val, dict):
                    safe_key = key.replace("drop_aug_", "drop_")
                    metrics[f"{safe_key}_auc_pr"] = float(val.get("auc_pr", 0))
                    metrics[f"{safe_key}_delta"]  = float(val.get("delta", 0))
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    return results




def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    require_inputs({
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "augmentation_ablation.json",
        args.force, "Stage 7.8 augmentation ablation"
    ):
        return
    augmentation_ablation(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())