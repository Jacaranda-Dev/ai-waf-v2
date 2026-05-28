"""
stages/3_data_augmentation/15_augmentation_probe.py
----------------------------------------------------
Enhanced Augmentation Probe  (replaces original LogReg/TF-IDF probe)

Improvements over the original:
  1. Lightweight 1D-CNN probe — better proxy for Transformer generalisation
     than Logistic Regression on TF-IDF features (captures n-gram position signals)
  2. Feature saturation analysis — per-class AUC-PR vs. sample-count curves;
     emits STOP_AUGMENTATION signal per class when marginal gains < threshold
  3. Real-only vs. augmented comparison still retained
  4. Incremental probing: evaluates at 25%, 50%, 75%, 100% of each class's
     augmented samples to build the saturation curve

Run:
    python stages/3_data_augmentation/15_augmentation_probe.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)

SATURATION_DELTA  = 0.002   # AUC-PR gain below this is considered saturated
SATURATION_WINDOW = 2       # consecutive checkpoints below delta → STOP signal


# ─────────────────────────────────────────────────────────────────────────────
# 1D-CNN probe  (character n-gram hashed features → 1D-CNN → binary classifier)
# ─────────────────────────────────────────────────────────────────────────────

class CharCNNProbe(nn.Module):
    """
    Lightweight 1D-CNN operating on hashed char-ngram feature vectors.

    Why CNN over LogReg:
      - 1D convolution over the feature space captures local n-gram patterns
        and their relative positions, a better analogue to the positional
        attention mechanism in the target 99M Transformer than a linear model.
      - Still fast enough for a probe (< 60s on CPU for 100k samples).
    """

    def __init__(self, n_features: int = 65_536, n_filters: int = 128, kernel_size: int = 5):
        super().__init__()
        # Reshape the flat hash vector into a 2D sequence for 1D conv
        # We treat blocks of `kernel_size` features as one "token position"
        self.n_features  = n_features
        self.kernel_size = kernel_size
        self.seq_len     = n_features // kernel_size  # e.g. 65536 // 5 ≈ 13107

        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(n_filters, n_filters // 2, kernel_size=5, padding=2)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.fc    = nn.Linear(n_filters // 2, 1)
        self.drop  = nn.Dropout(0.3)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_features] → [B, 1, n_features]
        x = x.unsqueeze(1)
        x = self.act(self.conv1(x))   # [B, n_filters, n_features]
        x = self.drop(x)
        x = self.act(self.conv2(x))   # [B, n_filters//2, n_features]
        x = self.pool(x).squeeze(-1)  # [B, n_filters//2]
        return self.fc(x).squeeze(-1) # [B]


def _vectorize(texts: list[str], n_features: int = 65_536) -> np.ndarray:
    """Hashed char-ngram features (1-4 grams) → dense float32 array."""
    vec = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        n_features=n_features,
        dtype=np.float32,
        norm="l2",
        alternate_sign=False,
    )
    return vec.transform(texts).toarray()  # type: ignore[return-value]


def _train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs:  int = 5,
    batch:   int = 256,
    device:  str = "cpu",
) -> CharCNNProbe:
    model = CharCNNProbe(n_features=X_train.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.BCEWithLogitsLoss()

    ds     = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        log.debug(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    return model


def _eval_probe(
    model:    CharCNNProbe,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    batch:    int = 512,
    device:   str = "cpu",
) -> dict:
    model.eval()
    ds     = TensorDataset(torch.from_numpy(X_val))
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    logits = []
    with torch.no_grad():
        for (xb,) in loader:
            logits.append(model(xb.to(device)).cpu().numpy())
    proba = torch.sigmoid(torch.from_numpy(np.concatenate(logits))).numpy()
    preds = (proba >= 0.5).astype(int)

    auc_pr = float(average_precision_score(y_val, proba))
    acc    = float((preds == y_val).mean())
    tp     = int(((preds == 1) & (y_val == 1)).sum())
    fp     = int(((preds == 1) & (y_val == 0)).sum())
    fn     = int(((preds == 0) & (y_val == 1)).sum())
    prec   = tp / max(1, tp + fp)
    rec    = tp / max(1, tp + fn)
    f1     = 2 * prec * rec / max(1e-9, prec + rec)

    return {"auc_pr": auc_pr, "f1": round(f1, 5), "precision": round(prec, 5),
            "recall": round(rec, 5), "accuracy": round(acc, 5)}


# ─────────────────────────────────────────────────────────────────────────────
# Saturation analysis
# ─────────────────────────────────────────────────────────────────────────────

def _saturation_check(
    aug_df:      pd.DataFrame,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    X_real:      np.ndarray,
    y_real:      np.ndarray,
    device:      str,
) -> dict[str, dict]:
    """
    For each attack class, probe at 25/50/75/100% of augmented samples.
    Flag STOP_AUGMENTATION if the last two AUC-PR increments are < SATURATION_DELTA.
    """
    results: dict[str, dict] = {}
    checkpoints = [0.25, 0.50, 0.75, 1.00]

    for attack_class in aug_df["attack_class"].unique():
        if attack_class == "benign":
            continue

        cls_df    = aug_df[aug_df["attack_class"] == attack_class]
        texts_cls = cls_df["raw"].tolist()
        labels_cls = cls_df["label"].to_numpy()

        if len(texts_cls) < 20:
            continue

        curve: list[dict] = []
        for frac in checkpoints:
            n = max(10, int(len(texts_cls) * frac))
            X_aug_sub = _vectorize(texts_cls[:n])
            y_aug_sub = labels_cls[:n]

            X_train = np.vstack([X_real, X_aug_sub])
            y_train = np.concatenate([y_real, y_aug_sub])

            model   = _train_probe(X_train, y_train, epochs=3, device=device)
            metrics = _eval_probe(model, X_val, y_val, device=device)
            curve.append({"frac": frac, "n": n, "auc_pr": metrics["auc_pr"]})
            log.info(f"  {attack_class} @ {int(frac*100)}%  n={n:,}  AUC-PR={metrics['auc_pr']:.4f}")

        # Saturation decision
        deltas  = [curve[i]["auc_pr"] - curve[i-1]["auc_pr"] for i in range(1, len(curve))]
        below   = [d < SATURATION_DELTA for d in deltas]
        saturated = sum(1 for b in below[-SATURATION_WINDOW:] if b) >= SATURATION_WINDOW

        results[attack_class] = {
            "curve":               curve,
            "deltas":              [round(d, 6) for d in deltas],
            "saturated":           saturated,
            "stop_augmentation":   saturated,
            "final_auc_pr":        curve[-1]["auc_pr"],
        }
        status = "STOP" if saturated else "CONTINUE"
        log.info(f"  {attack_class} saturation → {status}  (deltas={[round(d,5) for d in deltas]})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    splits_dir = Path(cfg.paths.data_splits)

    if not (splits_dir / "train.parquet").exists():
        log.error("train.parquet not found — run stratified split first")
        return
    if not (splits_dir / "val.parquet").exists():
        log.error("val.parquet not found — run stratified split first")
        return

    log.info(f"Augmentation probe running on: {device}")

    val_df  = pd.read_parquet(splits_dir / "val.parquet", columns=["raw", "label"])
    X_val   = _vectorize(val_df["raw"].tolist())
    y_val   = val_df["label"].to_numpy()

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw", "label", "source", "attack_class"])

    # ── 1. Real-only baseline ─────────────────────────────────────────────
    log.info("Training CNN probe on real-only data...")
    real_df   = train_df[~train_df["source"].str.startswith("aug")]
    X_real    = _vectorize(real_df["raw"].tolist())
    y_real    = real_df["label"].to_numpy()
    m_real    = _eval_probe(_train_probe(X_real, y_real, device=device), X_val, y_val, device=device)
    log.info(f"Real-only   : F1={m_real['f1']:.4f}  AUC-PR={m_real['auc_pr']:.4f}")

    # ── 2. Full (real + augmented) ────────────────────────────────────────
    log.info("Training CNN probe on real + augmented data...")
    X_full = _vectorize(train_df["raw"].tolist())
    y_full = train_df["label"].to_numpy()
    m_full = _eval_probe(_train_probe(X_full, y_full, device=device), X_val, y_val, device=device)
    log.info(f"Real+Aug    : F1={m_full['f1']:.4f}  AUC-PR={m_full['auc_pr']:.4f}")

    delta = m_full["auc_pr"] - m_real["auc_pr"]
    verdict = "augmentation helps" if delta > 0 else "augmentation may hurt — investigate"
    log.info(f"Delta AUC-PR: {delta:+.4f}  ({verdict})")

    # ── 3. Per-class saturation analysis ──────────────────────────────────
    log.info("Running per-class saturation analysis...")
    aug_df       = train_df[train_df["source"].str.startswith("aug")]
    saturation   = _saturation_check(aug_df, X_val, y_val, X_real, y_real, device)

    # ── 4. Write results ──────────────────────────────────────────────────
    results = {
        "probe_model":      "CharCNN(1D-CNN, char-ngram-hashed)",
        "real_only":        m_real,
        "real_and_aug":     m_full,
        "delta_auc_pr":     round(delta, 5),
        "verdict":          verdict,
        "saturation":       saturation,
        "stop_signals":     [cls for cls, s in saturation.items() if s["stop_augmentation"]],
    }
    out = Path(cfg.paths.reports) / "metrics" / "augmentation_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Probe results written → {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="05_augmentation_probe"):
            mlflow.log_params({
                "probe_model":    results["probe_model"],
                "n_stop_signals": len(results["stop_signals"]),
                "stop_classes":   results["stop_signals"],
            })
            log_metrics_dict({
                "real_only_auc_pr":  float(m_real["auc_pr"]),
                "real_only_f1":      float(m_real["f1"]),
                "real_aug_auc_pr":   float(m_full["auc_pr"]),
                "real_aug_f1":       float(m_full["f1"]),
                "delta_auc_pr":      float(delta),
            })
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    if results["stop_signals"]:
        log.warning(
            f"STOP_AUGMENTATION signals for: {results['stop_signals']} "
            f"(marginal AUC-PR gain < {SATURATION_DELTA} for {SATURATION_WINDOW} consecutive checkpoints)"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())