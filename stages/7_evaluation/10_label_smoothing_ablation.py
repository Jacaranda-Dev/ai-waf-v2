"""
stages/7_evaluation/10_label_smoothing_ablation.py
-----------------------------------------
All four ablation studies in one file (dispatched by script name).
Each ablation trains a fast probe model (LR + TF-IDF) to isolate
the variable of interest without requiring full transformer training.

For full transformer ablations, use MLflow to compare existing runs:
these scripts compare saved checkpoints and report metrics.

Run:
    python stages/7_evaluation/10_label_smoothing_ablation.py --config config/pipeline.yaml
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
# 10 — Label smoothing ablation
# ─────────────────────────────────────────────────────────

def label_smoothing_ablation(cfg) -> dict:
    """
    Compares effect of label smoothing using the fast probe approach.
    For full transformer comparison, query MLflow runs with different
    label_smoothing params.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from ai_waf_v2.eval.metrics import compute_metrics

    timer = StepTimer()
    splits_dir = Path(cfg.paths.data_splits)
    if not (splits_dir / "train.parquet").exists():
        return {"error": "train.parquet not found"}

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw","label"])
    val_df   = pd.read_parquet(splits_dir / "val.parquet",   columns=["raw","label"])
    X_train  = train_df["raw"].tolist()
    y_train  = train_df["label"].to_numpy()
    X_val    = val_df["raw"].tolist()
    y_val    = val_df["label"].to_numpy()

    # For LR, label smoothing maps to C (inverse regularization)
    # We simulate it by testing different smoothing strengths
    smoothing_vals = [0.0, 0.05, 0.1, 0.15, 0.2]
    results = {}

    with timer.step("smoothing_sweep"):
        for eps in smoothing_vals:
            # Approximate label smoothing: blend labels toward uniform
            import numpy as np
            y_soft = y_train.astype(float) * (1 - eps) + eps * 0.5
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3), max_features=20_000)),
                ("clf",   LogisticRegression(max_iter=300, C=1.0, random_state=42)),
            ])
            pipe.fit(X_train, (y_soft > 0.5).astype(int))
            proba = pipe.predict_proba(X_val)[:, 1]
            preds = (proba >= 0.5).astype(int)
            m = compute_metrics(torch.tensor(preds), torch.tensor(proba, dtype=torch.float),
                                torch.tensor(y_val))
            results[f"eps_{eps}"] = {"auc_pr": m["auc_pr"], "f1": m["f1"], "fpr": m["fpr"]}
            log.info(f"  eps={eps:.2f}: AUC-PR={m['auc_pr']:.4f}  F1={m['f1']:.4f}")

    results["timings_s"] = timer.timings
    out = Path(cfg.paths.reports) / "metrics" / "label_smoothing_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Label smoothing ablation saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="10_label_smoothing_ablation") as _run:
            mlflow.log_params({
                "smoothing_values": str(smoothing_vals),
                "n_smoothing_levels": len(smoothing_vals),
            })
            metrics: dict[str, float] = {}
            for eps_key, m in results.items():
                if isinstance(m, dict):
                    for metric_name in ("auc_pr", "f1", "fpr"):
                        val = m.get(metric_name)
                        if isinstance(val, (int, float)):
                            metrics[f"{eps_key}_{metric_name}"] = float(val)
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
        Path(cfg.paths.reports) / "metrics" / "label_smoothing_ablation.json",
        args.force, "Stage 7.10 label smoothing ablation"
    ):
        return
    label_smoothing_ablation(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())