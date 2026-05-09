"""
stages/2_baselines/05_tfidf_lightgbm.py
----------------------------------------
Stage 0.3 — TF-IDF char n-gram (1–3) + LightGBM baseline.
Mirrors 02_tfidf_xgboost.py but uses LightGBM for comparison.

Run:
    python stages/2_baselines/05_tfidf_lightgbm.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    splits_dir  = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw","label"])
    val_df   = pd.read_parquet(splits_dir / "val.parquet",   columns=["raw","label"])
    test_df  = pd.read_parquet(splits_dir / "test.parquet",  columns=["raw","label"])

    X_train, y_train = train_df["raw"].tolist(), train_df["label"].to_numpy()
    X_val,   y_val   = val_df["raw"].tolist(),   val_df["label"].to_numpy()
    X_test,  y_test  = test_df["raw"].tolist(),  test_df["label"].to_numpy()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1,3),
            max_features=50_000, sublinear_tf=True,
        )),
        ("clf", LGBMClassifier(
            n_estimators=500, max_depth=6,
            learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=cfg.project.seed,
            n_jobs=-1, verbose=-1,
        )),
    ])

    log.info("Training LightGBM baseline...")
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    log.info(f"Done in {train_time:.1f}s")

    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = pipe.predict_proba(X)[:,1]
        preds = (proba >= 0.5).astype(int)
        m = compute_metrics(
            torch.tensor(preds), torch.tensor(proba, dtype=torch.float),
            torch.tensor(y),
        )
        log.info(f"{split_name}: F1={m['f1']:.4f}  FPR={m['fpr']:.5f}  AUC-PR={m['auc_pr']:.4f}")

    # Latency
    lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        pipe.predict_proba(X_test[:64])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)

    test_proba = pipe.predict_proba(X_test)[:,1]
    test_preds = (test_proba >= 0.5).astype(int)
    test_m = compute_metrics(
        torch.tensor(test_preds),
        torch.tensor(test_proba, dtype=torch.float),
        torch.tensor(y_test),
    )

    result = {
        "train_time_s": round(train_time, 2),
        "overall": test_m,
        "latency": {
            "batch_size": 64,
            "p50_ms": round(float(np.percentile(lat,50)),3),
            "p95_ms": round(float(np.percentile(lat,95)),3),
            "p99_ms": round(float(np.percentile(lat,99)),3),
            "throughput_rps": round(64/(lat.mean()/1000),1),
            "device": "cpu",
        },
    }

    existing = json.loads((reports_dir/"baselines.json").read_text()) \
        if (reports_dir/"baselines.json").exists() else {}
    existing["lightgbm_tfidf"] = result
    (reports_dir/"baselines.json").write_text(json.dumps(existing, indent=2))
    log.info(f"LightGBM baseline saved.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())