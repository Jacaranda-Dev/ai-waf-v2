"""Stage 2.7 — Measure augmentation contribution via a fast probe model."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def _probe(X_train, y_train, X_val, y_val):
    pipe = Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3), max_features=20000)),
                     ("clf",  LogisticRegression(max_iter=500, C=1.0, random_state=42))])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:,1]
    preds = (proba >= 0.5).astype(int)
    return compute_metrics(torch.tensor(preds), torch.tensor(proba,dtype=torch.float), torch.tensor(y_val))
def run(args):
    configure_root(); cfg = load_config(args.config)
    splits_dir = Path(cfg.paths.data_splits)
    if not (splits_dir/"train.parquet").exists(): log.error("No splits found"); return
    val_df  = pd.read_parquet(splits_dir/"val.parquet", columns=["raw","label"])
    X_val, y_val = val_df["raw"].tolist(), val_df["label"].to_numpy()
    # Real-only
    real_df = pd.read_parquet(splits_dir/"train.parquet", columns=["raw","label","source"])
    real_only = real_df[~real_df["source"].str.startswith("aug")]
    m_real = _probe(real_only["raw"].tolist(), real_only["label"].to_numpy(), X_val, y_val)
    log.info(f"Real-only   : F1={m_real[\'f1\']:.4f}  AUC-PR={m_real[\'auc_pr\']:.4f}")
    # Full (real + augmented)
    m_full = _probe(real_df["raw"].tolist(), real_df["label"].to_numpy(), X_val, y_val)
    log.info(f"Real+Aug    : F1={m_full[\'f1\']:.4f}  AUC-PR={m_full[\'auc_pr\']:.4f}")
    delta = m_full["auc_pr"] - m_real["auc_pr"]
    log.info(f"Delta AUC-PR: {delta:+.4f}  ({'positive — augmentation helps' if delta > 0 else 'negative — augmentation may hurt'})")
    out = Path(cfg.paths.reports)/"metrics"/"augmentation_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"real_only": m_real, "real_and_aug": m_full, "delta_auc_pr": round(delta,5)}, indent=2))
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
