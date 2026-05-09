"""
stages/2_baselines/04_tfidf_xgboost.py
---------------------------------------
Stage 0.2 — TF-IDF char n-gram (1-3) + XGBoost baseline.

Mirrors 05_tfidf_lightgbm.py; together they form the classical-ML
baseline pair.  Three meaningful differences vs the LightGBM stage:

  1. GPU-aware tree method — uses CUDA when available, falls back to
     CPU hist gracefully.  XGBoost >= 2.0 syntax (device=) with a
     fallback for older installs (tree_method="gpu_hist").

  2. Early stopping on val AUC-PR — TF-IDF is fitted first so a
     transformed eval_set can be passed to XGBoost.  Training stops
     when aucpr has not improved for `early_stopping_rounds` rounds,
     preventing overfitting on large augmented datasets.

  3. scale_pos_weight — computed from the training label distribution
     so the classifier accounts for benign/malicious imbalance without
     requiring a manually tuned class weight.

Run:
    python stages/2_baselines/04_tfidf_xgboost.py --config config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)

# ─── hyper-parameters ────────────────────────────────────────────────────────
TFIDF_KWARGS = dict(
    analyzer="char_wb",
    ngram_range=(1, 3),
    max_features=50_000,
    sublinear_tf=True,
)

XGB_KWARGS = dict(
    n_estimators=1_000,        # upper bound; early stopping will cut this
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,        # regularisation; helps on imbalanced splits
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="aucpr",       # early-stop on AUC-PR, not AUC-ROC
    early_stopping_rounds=50,
    n_jobs=-1,
    verbosity=0,
)

LATENCY_BATCH   = 64
LATENCY_WARMUP  = 10
LATENCY_RUNS    = 50


# ─── GPU detection ───────────────────────────────────────────────────────────

def _xgb_device_kwargs() -> dict:
    """
    Return XGBoost device kwargs that work across versions.

    XGBoost >= 2.0  : tree_method="hist", device="cuda"|"cpu"
    XGBoost <  2.0  : tree_method="gpu_hist"|"hist"
    """
    cuda = torch.cuda.is_available()

    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if xgb_version >= (2, 0):
        return {"tree_method": "hist", "device": "cuda" if cuda else "cpu"}
    else:
        return {"tree_method": "gpu_hist" if cuda else "hist"}


# ─── main ────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    splits_dir  = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── load splits ──────────────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.parquet"
        if not path.exists():
            log.error(f"{path} not found — run data stages first")
            return

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw", "label"])
    val_df   = pd.read_parquet(splits_dir / "val.parquet",   columns=["raw", "label"])
    test_df  = pd.read_parquet(splits_dir / "test.parquet",  columns=["raw", "label", "attack_class"])

    X_train, y_train = train_df["raw"].tolist(), train_df["label"].to_numpy()
    X_val,   y_val   = val_df["raw"].tolist(),   val_df["label"].to_numpy()
    X_test,  y_test  = test_df["raw"].tolist(),  test_df["label"].to_numpy()
    test_classes      = test_df["attack_class"].tolist()

    # ── class imbalance weight ───────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        log.error("Training set contains no positive (malicious) samples.")
        return
    scale_pos_weight = n_neg / n_pos
    log.info(
        f"Label distribution — benign: {n_neg:,}  malicious: {n_pos:,}  "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ── build pipeline objects ───────────────────────────────────────────────
    # TF-IDF and XGBoost are kept as named sklearn Pipeline steps so that
    # pipe.predict_proba(X) works for latency benchmarking after manual fit.
    device_kwargs = _xgb_device_kwargs()
    log.info(f"XGBoost device kwargs: {device_kwargs}")

    tfidf = TfidfVectorizer(**TFIDF_KWARGS)
    clf   = xgb.XGBClassifier(
        **XGB_KWARGS,
        **device_kwargs,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.project.seed,
    )
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])

    # ── fit — TF-IDF first so we can pass a transformed eval_set ─────────────
    # Fitting TF-IDF separately is necessary because sklearn Pipeline does NOT
    # transform the eval_set through earlier steps when clf__eval_set is passed.
    # Since pipe stores references, calling tfidf.fit_transform() here means
    # pipe.predict_proba() will use the already-fitted vectoriser correctly.
    log.info("Fitting TF-IDF vectoriser...")
    t0 = time.perf_counter()
    X_train_t = tfidf.fit_transform(X_train)
    X_val_t   = tfidf.transform(X_val)
    tfidf_time = time.perf_counter() - t0
    log.info(
        f"TF-IDF done in {tfidf_time:.1f}s — "
        f"vocab size: {len(tfidf.vocabulary_):,}"
    )

    log.info("Training XGBoost (early stopping on val aucpr)...")
    t0 = time.perf_counter()
    clf.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    train_time = time.perf_counter() - t0

    best_iter = clf.best_iteration
    best_score = clf.best_score
    log.info(
        f"Training done in {train_time:.1f}s — "
        f"best iteration: {best_iter}  "
        f"best val aucpr: {best_score:.4f}"
    )

    # ── val / test evaluation ─────────────────────────────────────────────────
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = pipe.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        m = compute_metrics(
            torch.tensor(preds),
            torch.tensor(proba, dtype=torch.float),
            torch.tensor(y),
        )
        log.info(
            f"{split_name}: F1={m['f1']:.4f}  "
            f"FPR={m['fpr']:.5f}  "
            f"AUC-PR={m['auc_pr']:.4f}  "
            f"AUC-ROC={m['auc_roc']:.4f}"
        )

    # ── per-class breakdown on test ───────────────────────────────────────────
    from ai_waf_v2.eval.metrics import compute_per_class_metrics
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)
    test_m     = compute_metrics(
        torch.tensor(test_preds),
        torch.tensor(test_proba, dtype=torch.float),
        torch.tensor(y_test),
    )
    per_class = compute_per_class_metrics(
        torch.tensor(test_preds),
        torch.tensor(test_proba, dtype=torch.float),
        torch.tensor(y_test),
        test_classes,
    )

    # ── latency benchmark ─────────────────────────────────────────────────────
    # Warm up first (JIT / cache effects), then measure.
    probe = X_test[:LATENCY_BATCH]
    for _ in range(LATENCY_WARMUP):
        pipe.predict_proba(probe)

    lat_ms: list[float] = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        pipe.predict_proba(probe)
        lat_ms.append((time.perf_counter() - t0) * 1_000)
    lat = np.array(lat_ms)

    device_used = device_kwargs.get("device", "cuda" if "gpu" in device_kwargs.get("tree_method","") else "cpu")
    log.info(
        f"Latency (batch={LATENCY_BATCH}, {LATENCY_RUNS} runs) — "
        f"p50={np.percentile(lat,50):.1f}ms  "
        f"p95={np.percentile(lat,95):.1f}ms  "
        f"p99={np.percentile(lat,99):.1f}ms  "
        f"device={device_used}"
    )

    # ── feature importance (top 20) ───────────────────────────────────────────
    # XGBoost gain importance; maps back to char n-gram strings via TF-IDF.
    feature_names   = tfidf.get_feature_names_out()
    importance_vals = clf.feature_importances_          # gain, shape (n_features,)
    top_idx         = np.argsort(importance_vals)[::-1][:20]
    top_features    = [
        {"ngram": str(feature_names[i]), "importance": round(float(importance_vals[i]), 6)}
        for i in top_idx
    ]

    feat_path = reports_dir / "xgboost_top_features.json"
    feat_path.write_text(json.dumps(top_features, indent=2))
    log.info(f"Top-20 features saved to {feat_path}")
    log.info(
        "Top 5 char n-grams by gain: "
        + "  ".join(f['ngram'] for f in top_features[:5])
    )

    # ── write result to shared baselines.json ────────────────────────────────
    result = {
        "tfidf_time_s":   round(tfidf_time, 2),
        "train_time_s":   round(train_time, 2),
        "best_iteration": best_iter,
        "best_val_aucpr": round(best_score, 6),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "overall":        test_m,
        "per_class":      per_class,
        "latency": {
            "batch_size":    LATENCY_BATCH,
            "n_runs":        LATENCY_RUNS,
            "p50_ms":        round(float(np.percentile(lat, 50)), 3),
            "p95_ms":        round(float(np.percentile(lat, 95)), 3),
            "p99_ms":        round(float(np.percentile(lat, 99)), 3),
            "throughput_rps": round(LATENCY_BATCH / (lat.mean() / 1_000), 1),
            "device":        device_used,
        },
        "top_features_path": str(feat_path),
    }

    baselines_path = reports_dir / "baselines.json"
    existing = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    existing["xgboost_tfidf"] = result
    baselines_path.write_text(json.dumps(existing, indent=2))
    log.info(f"XGBoost baseline saved to {baselines_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 0.2 — TF-IDF + XGBoost baseline"
    )
    p.add_argument("--config", default="config/pipeline.yaml",
                   help="Path to pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())