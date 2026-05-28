"""
stages/2_baselines/03_classical_ml_baseline.py
-----------------------------------------------
Stage 2.2 — Classical ML and Fast-Match baselines.

Replaces and consolidates:
    03_tfidf_xgboost.py    (XGBoost, TF-IDF)
    04_tfidf_lightgbm.py   (LightGBM, TF-IDF — architecturally redundant)

Three distinct baselines in one script, each providing qualitatively
different information for the research paper:

  A. Aho-Corasick Fast-Match
     Pure string matching on a compiled keyword automaton — the approach
     used by the first layer of commercial WAFs (Cloudflare, AWS WAF).
     Establishes the *maximum possible throughput* upper bound and the
     *minimum possible accuracy* lower bound for any signature approach.

  B. HashedNGram + XGBoost  (primary classical-ML baseline)
     Replaces TF-IDF with sklearn's HashingVectorizer (fixed 2²⁰ buckets,
     no vocabulary stored) to simulate production memory constraints.
     Uses GPU-accelerated XGBoost with early stopping on val AUC-PR,
     scale_pos_weight for imbalance, and HTTP field-aware feature
     engineering (see _http_features()).

  C. HashedNGram + XGBoost  (feature-ablation variant)
     Identical model trained WITHOUT structural HTTP features, so the paper
     can quantify exactly how much lift comes from field-aware features vs
     raw n-gram statistics.

LightGBM is intentionally dropped: it and XGBoost are both gradient-
boosted decision trees over the same feature space, and running both adds
wall-clock time without distinct qualitative insight.  If you need to
compare boosting libraries, pass --also-lgbm; the result is appended to
baselines.json without cluttering the default run.

SLO Audit
──────────
All three baselines read slos.json and emit PASS/FAIL verdicts per metric.
The expected outcome for a research paper is:

  Fast-Match  : Latency PASS / Accuracy FAIL  → motivates ML over regex
  XGBoost     : Accuracy close-to-PASS / Latency FAIL → motivates Transformer
  XGBoost+feat: same story, higher accuracy ceiling

Run:
    python stages/2_baselines/03_classical_ml_baseline.py --config config/pipeline.yaml

    # also run the LightGBM comparison (adds ~2 min, appended to baselines.json):
    python ... --also-lgbm

    # skip the feature-ablation run (faster):
    python ... --no-ablation
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xgboost as xgb
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from ai_waf_v2.eval.metrics import compute_metrics, compute_per_class_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

LATENCY_WARMUP = 10
LATENCY_RUNS   = 50
LATENCY_BATCH  = 64

# HashingVectorizer bucket count — 2²⁰ ≈ 1M features, ~4 MB RAM
# (vs TF-IDF which stores a vocabulary dict of potentially 50 k+ strings)
HASH_N_FEATURES = 2 ** 20

XGB_BASE_KWARGS = dict(
    n_estimators          = 1_000,   # upper bound; early stopping cuts this
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    gamma                 = 0.1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    eval_metric           = "aucpr",
    early_stopping_rounds = 50,
    n_jobs                = -1,
    verbosity             = 0,
)


# ─────────────────────────────────────────────────────────
# Aho-Corasick Fast-Match baseline
# ─────────────────────────────────────────────────────────

# High-confidence attack signatures — deliberately kept compact so the
# automaton compiles in <1 s.  In a production WAF this list is 10-100×
# longer; using a subset here correctly understates accuracy and overstates
# throughput, which is the conservative bound we want for the paper.
_AC_SIGNATURES: list[str] = [
    # SQLi
    "union select", "union all select", "' or '1'='1", "or 1=1",
    "insert into", "drop table", "exec(", "execute(", "xp_cmdshell",
    "sleep(", "benchmark(", "waitfor delay",
    # XSS
    "<script", "javascript:", "onerror=", "onload=", "document.cookie",
    "alert(", "eval(",
    # LFI / path traversal
    "../", "..\\", "/etc/passwd", "/etc/shadow", "php://filter",
    # RFI
    "http://", "ftp://",
    # SSRF
    "169.254.169.254", "metadata.google.internal", "file://", "gopher://",
    # CMDi
    "; id", "| whoami", "&& cat /", "`id`", "$(id)",
    # XXE
    "<!entity", "<!doctype",
    # SSTI
    "{{", "}}", "{%", "%}",
]


def _build_ac_matcher() -> Callable[[str], int]:
    """
    Build an Aho-Corasick automaton if the `ahocorasick` package is
    available; fall back to a plain Python multi-string search otherwise.

    Returns a callable: raw_text → 1 (match) | 0 (no match).
    The fallback is ~10× slower but produces identical predictions.
    """
    try:
        import ahocorasick
        A = ahocorasick.Automaton()
        for idx, sig in enumerate(_AC_SIGNATURES):
            A.add_word(sig.lower(), (idx, sig))
        A.make_automaton()

        def _match_ac(text: str) -> int:
            t = text.lower()
            for _ in A.iter(t):
                return 1
            return 0

        log.info(
            f"Aho-Corasick automaton compiled "
            f"({len(_AC_SIGNATURES)} signatures)"
        )
        return _match_ac

    except ImportError:
        log.warning(
            "pyahocorasick not installed — using plain Python fallback. "
            "Install with: pip install pyahocorasick"
        )
        sigs_lower = [s.lower() for s in _AC_SIGNATURES]

        def _match_py(text: str) -> int:
            t = text.lower()
            return 1 if any(s in t for s in sigs_lower) else 0

        return _match_py


def _run_fast_match(
    raws:    list[str],
    labels:  list[int],
    classes: list[str],
) -> tuple[dict, dict]:
    """Evaluate the Aho-Corasick baseline; return (metrics, latency)."""
    matcher = _build_ac_matcher()

    t0    = time.perf_counter()
    preds = [matcher(r) for r in raws]
    rps   = len(raws) / (time.perf_counter() - t0)

    preds_t  = torch.tensor(preds,  dtype=torch.long)
    probs_t  = preds_t.float()
    labels_t = torch.tensor(labels, dtype=torch.long)

    metrics   = compute_metrics(preds_t, probs_t, labels_t)
    per_class = compute_per_class_metrics(preds_t, probs_t, labels_t, classes)

    # Latency: measure per-request time (AC is per-request, not batched)
    probe_raw = raws[:1]
    for _ in range(LATENCY_WARMUP):
        matcher(probe_raw[0])
    lat_us = []
    for r in raws[:500]:            # sample 500 requests
        t0 = time.perf_counter()
        matcher(r)
        lat_us.append((time.perf_counter() - t0) * 1_000_000)
    lat_us_arr = np.array(lat_us)

    latency = {
        "batch_size":     1,
        "n_signatures":   len(_AC_SIGNATURES),
        "p50_us":         round(float(np.percentile(lat_us_arr, 50)), 3),
        "p99_us":         round(float(np.percentile(lat_us_arr, 99)), 3),
        "p99_ms":         round(float(np.percentile(lat_us_arr, 99)) / 1_000, 4),
        "throughput_rps": round(rps, 0),
        "device":         "cpu",
    }

    log.info(
        f"Fast-Match: F1={metrics['f1']:.4f}  "
        f"FPR={metrics['fpr']:.5f}  "
        f"Recall={metrics['recall']:.4f}  "
        f"RPS={rps:,.0f}  "
        f"p99={latency['p99_us']:.1f}µs"
    )

    return {
        "model":     "aho_corasick_fast_match",
        "overall":   metrics,
        "per_class": per_class,
        "latency":   latency,
    }, metrics, latency


# ─────────────────────────────────────────────────────────
# HTTP field-aware features
# ─────────────────────────────────────────────────────────

# Patterns that are dangerous in request fields but near-harmless in raw text.
_FIELD_PATTERNS = {
    "has_sqli_in_query":  re.compile(r"[?&][^=]+=.*(?:union\s+select|or\s+1=1|'--)", re.I),
    "has_sqli_in_body":   re.compile(r"(?:^|\r?\n)(?:.*=.*(?:union\s+select|' or '))", re.I),
    "has_xss_in_ua":      re.compile(r"user-agent:.*<\s*script", re.I),
    "has_path_traversal": re.compile(r"(?:GET|POST|PUT)\s+[^\s]*(?:\.\.[\\/]){2,}", re.I),
    "has_encoded_payload":re.compile(r"%(?:27|3c|3e|22|2e%2e){2,}", re.I),
    "has_long_value":     None,   # computed numerically
    "has_many_params":    None,   # computed numerically
    "method_is_rare":     None,   # PUT/DELETE/PATCH in benign traffic is uncommon
}

_RARE_METHODS = re.compile(r"^(?:PUT|DELETE|PATCH|OPTIONS|TRACE|CONNECT)\s", re.I)


def _http_features(raw: str) -> list[float]:
    """
    Extract a small vector of structural HTTP features from the raw string.

    These features capture context that character n-grams cannot: e.g. that
    a single-quote in the User-Agent header is almost always benign, whereas
    the same character in a query-string value is a SQLi signal.

    Returns a list[float] of length 7 (one per feature above).
    """
    q  = re.search(r"\?([^\s#]+)", raw)
    qs = q.group(1) if q else ""
    n_params = qs.count("&") + (1 if qs else 0)

    vals = [
        float(bool(_FIELD_PATTERNS["has_sqli_in_query"].search(raw))),
        float(bool(_FIELD_PATTERNS["has_sqli_in_body"].search(raw))),
        float(bool(_FIELD_PATTERNS["has_xss_in_ua"].search(raw))),
        float(bool(_FIELD_PATTERNS["has_path_traversal"].search(raw))),
        float(bool(_FIELD_PATTERNS["has_encoded_payload"].search(raw))),
        float(min(len(raw), 10_000) / 10_000),     # Normalized request length
        float(n_params > 10),                       # unusually many query params
    ]
    return vals


N_HTTP_FEATURES = 7    # must match length of _http_features() output


# ─────────────────────────────────────────────────────────
# XGBoost device detection
# ─────────────────────────────────────────────────────────

def _xgb_device_kwargs() -> dict:
    cuda = torch.cuda.is_available()
    version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if version >= (2, 0):
        return {"tree_method": "hist", "device": "cuda" if cuda else "cpu"}
    return {"tree_method": "gpu_hist" if cuda else "hist"}


# ─────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────

def _build_features(
    texts:           list[str],
    vectorizer:      HashingVectorizer,
    include_http:    bool,
    fit:             bool = False,
) -> sp.spmatrix | np.ndarray:
    """
    Build the feature matrix for a list of raw HTTP strings.

    When include_http=True, appends the N_HTTP_FEATURES structural columns
    to the hashed n-gram sparse matrix via scipy hstack.
    When include_http=False, returns the bare n-gram matrix (ablation mode).
    """
    if fit:
        X_ng = vectorizer.fit_transform(texts)
    else:
        X_ng = vectorizer.transform(texts)

    if not include_http:
        return X_ng

    http_mat = np.array([_http_features(t) for t in texts], dtype=np.float32)
    http_sp  = sp.csr_matrix(http_mat)
    return sp.hstack([X_ng, http_sp], format="csr")


# ─────────────────────────────────────────────────────────
# SLO audit
# ─────────────────────────────────────────────────────────

def _load_slos(reports_dir: Path) -> dict | None:
    slo_path = reports_dir / "slos.json"
    if not slo_path.exists():
        log.warning("slos.json not found — SLO audit skipped")
        return None
    return json.loads(slo_path.read_text())


def _audit_slos(model_key: str, metrics: dict, latency: dict, slos: dict) -> dict[str, str]:
    """
    Emit a PASS/FAIL verdict for each SLO dimension.

    Latency target used is offline_p99_ms (batch=64 evaluation context).
    For Fast-Match the p99 is in microseconds; it is converted to ms before
    comparison so the same SLO threshold applies to all baselines.
    """
    verdicts: dict[str, str] = {}

    max_fpr          = slos["accuracy"]["max_false_positive_rate"]
    verdicts["fpr"]  = "PASS" if metrics.get("fpr", 1.0) <= max_fpr else "FAIL"

    p99_limit             = slos["latency"]["offline_p99_ms"]
    verdicts["latency_p99"] = (
        "PASS" if latency.get("p99_ms", float("inf")) <= p99_limit else "FAIL"
    )

    min_rps               = slos["latency"]["throughput_min_rps"]
    verdicts["throughput"] = (
        "PASS" if latency.get("throughput_rps", 0) >= min_rps else "FAIL"
    )

    for metric, verdict in verdicts.items():
        icon = "✓" if verdict == "PASS" else "✗"
        log.info(f"  SLO [{model_key}] {icon} {metric}: {verdict}")

    return verdicts


# ─────────────────────────────────────────────────────────
# XGBoost training + evaluation
# ─────────────────────────────────────────────────────────

def _run_xgboost(
    X_train:    sp.spmatrix,
    y_train:    np.ndarray,
    X_val:      sp.spmatrix,
    y_val:      np.ndarray,
    X_test:     sp.spmatrix,
    y_test:     np.ndarray,
    test_raw:   list[str],
    test_classes: list[str],
    vectorizer: HashingVectorizer,
    include_http: bool,
    seed:       int,
    model_key:  str,
) -> tuple[dict, dict, dict]:
    """
    Train XGBoost, evaluate on val + test, benchmark latency.
    Returns (result_dict, metrics, latency).
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("Training set has no positive (malicious) samples.")
    scale_pos_weight = n_neg / n_pos

    device_kwargs = _xgb_device_kwargs()
    clf = xgb.XGBClassifier(
        **XGB_BASE_KWARGS,
        **device_kwargs,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )

    log.info(f"Training XGBoost [{model_key}] — device={device_kwargs}  http_features={include_http}")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - t0
    log.info(
        f"  Done in {train_time:.1f}s — "
        f"best_iter={clf.best_iteration}  "
        f"best_val_aucpr={clf.best_score:.4f}"
    )

    # Evaluation
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)
    metrics    = compute_metrics(
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
    log.info(
        f"  Test: F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.5f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}"
    )

    # Latency benchmark: full pipeline (vectorise + predict) on a fixed probe
    # We re-vectorise the probe batch each run to include vectorizer cost,
    # since that cost is incurred in production on every request.
    probe_texts = test_raw[:LATENCY_BATCH]
    for _ in range(LATENCY_WARMUP):
        probe_X = _build_features(probe_texts, vectorizer, include_http)
        clf.predict_proba(probe_X)

    lat_ms: list[float] = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        probe_X = _build_features(probe_texts, vectorizer, include_http)
        clf.predict_proba(probe_X)
        lat_ms.append((time.perf_counter() - t0) * 1_000)

    lat = np.array(lat_ms)
    latency = {
        "batch_size":     LATENCY_BATCH,
        "n_runs":         LATENCY_RUNS,
        "p50_ms":         round(float(np.percentile(lat, 50)), 3),
        "p95_ms":         round(float(np.percentile(lat, 95)), 3),
        "p99_ms":         round(float(np.percentile(lat, 99)), 3),
        "throughput_rps": round(LATENCY_BATCH / (lat.mean() / 1_000), 1),
        "device":         device_kwargs.get("device", "cpu"),
    }
    log.info(
        f"  Latency p50={latency['p50_ms']}ms  "
        f"p99={latency['p99_ms']}ms  "
        f"rps={latency['throughput_rps']}"
    )

    result = {
        "model":             model_key,
        "include_http_features": include_http,
        "train_time_s":      round(train_time, 2),
        "best_iteration":    clf.best_iteration,
        "best_val_aucpr":    round(float(clf.best_score), 6),
        "scale_pos_weight":  round(scale_pos_weight, 4),
        "n_features":        X_train.shape[1],
        "overall":           metrics,
        "per_class":         per_class,
        "latency":           latency,
    }
    return result, metrics, latency


# ─────────────────────────────────────────────────────────
# Optional LightGBM comparison
# ─────────────────────────────────────────────────────────

def _run_lightgbm(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_val:   sp.spmatrix,
    y_val:   np.ndarray,
    X_test:  sp.spmatrix,
    y_test:  np.ndarray,
    test_raw: list[str],
    test_classes: list[str],
    vectorizer: HashingVectorizer,
    include_http: bool,
    seed: int,
) -> dict:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.error("lightgbm not installed — skipping LightGBM run")
        return {}

    clf = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    test_proba = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)
    metrics    = compute_metrics(
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

    lat_ms = []
    probe_texts = test_raw[:LATENCY_BATCH]
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        probe_X = _build_features(probe_texts, vectorizer, include_http)
        clf.predict_proba(probe_X)
        lat_ms.append((time.perf_counter() - t0) * 1_000)
    lat = np.array(lat_ms)

    latency = {
        "batch_size": LATENCY_BATCH,
        "p50_ms":     round(float(np.percentile(lat, 50)), 3),
        "p99_ms":     round(float(np.percentile(lat, 99)), 3),
        "throughput_rps": round(LATENCY_BATCH / (lat.mean() / 1_000), 1),
        "device": "cpu",
    }
    log.info(
        f"LightGBM: F1={metrics['f1']:.4f}  "
        f"FPR={metrics['fpr']:.5f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  "
        f"p99={latency['p99_ms']}ms"
    )
    return {
        "model":        "lightgbm_hashed_ngram",
        "train_time_s": round(train_time, 2),
        "overall":      metrics,
        "per_class":    per_class,
        "latency":      latency,
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    splits_dir  = Path(args.split_dir) if args.split_dir else Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ───────────────────────────────────────────────────────────
    for split in ("train", "val", "test"):
        p = splits_dir / f"{split}.parquet"
        if not p.exists():
            log.error(f"{p} not found — run data stages first")
            return

    train_df = pd.read_parquet(splits_dir / "train.parquet", columns=["raw", "label"])
    val_df   = pd.read_parquet(splits_dir / "val.parquet",   columns=["raw", "label"])
    test_df  = pd.read_parquet(splits_dir / "test.parquet",  columns=["raw", "label", "attack_class"])

    X_train_raw = train_df["raw"].tolist()
    y_train     = train_df["label"].to_numpy()
    X_val_raw   = val_df["raw"].tolist()
    y_val       = val_df["label"].to_numpy()
    X_test_raw  = test_df["raw"].tolist()
    y_test      = test_df["label"].to_numpy()
    test_classes = test_df["attack_class"].tolist()

    slos = _load_slos(reports_dir)

    baselines_path = reports_dir / args.out_file
    existing = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    phase = args.phase

    # ── Baseline A: Aho-Corasick Fast-Match ──────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("Baseline A: Aho-Corasick Fast-Match")
    log.info("═" * 60)
    result_ac, m_ac, lat_ac = _run_fast_match(X_test_raw, y_test.tolist(), test_classes)
    result_ac["slo_verdicts"] = _audit_slos("fast_match", m_ac, lat_ac, slos) if slos else {}
    result_ac["phase"] = phase
    existing["aho_corasick_fast_match"] = result_ac

    # ── Build shared hashed n-gram vectorizer (fit once on train) ─────────────
    log.info("\n" + "═" * 60)
    log.info("Building HashedNGram vectorizer (fit on train)…")
    log.info("═" * 60)
    vectorizer = HashingVectorizer(
        analyzer    = "char_wb",
        ngram_range = (1, 3),
        n_features  = HASH_N_FEATURES,
        norm        = "l2",
        alternate_sign = False,   # keep values non-negative for XGBoost
    )

    # Build all feature matrices once to avoid redundant vectorization
    log.info("Vectorizing train…")
    X_train_ng = vectorizer.fit_transform(X_train_raw)
    log.info("Vectorizing val…")
    X_val_ng   = vectorizer.transform(X_val_raw)
    log.info("Vectorizing test…")
    X_test_ng  = vectorizer.transform(X_test_raw)

    # Append structural HTTP features
    def _with_http(X_ng: sp.spmatrix, texts: list[str]) -> sp.spmatrix:
        http_mat = np.array([_http_features(t) for t in texts], dtype=np.float32)
        return sp.hstack([X_ng, sp.csr_matrix(http_mat)], format="csr")

    X_train_full = _with_http(X_train_ng, X_train_raw)
    X_val_full   = _with_http(X_val_ng,   X_val_raw)
    X_test_full  = _with_http(X_test_ng,  X_test_raw)

    # ── Baseline B: XGBoost + HashedNGram + HTTP features ────────────────────
    log.info("\n" + "═" * 60)
    log.info("Baseline B: XGBoost + HashedNGram + HTTP field features")
    log.info("═" * 60)
    result_xgb, m_xgb, lat_xgb = _run_xgboost(
        X_train_full, y_train, X_val_full, y_val, X_test_full, y_test,
        X_test_raw, test_classes, vectorizer,
        include_http=True, seed=cfg.project.seed,
        model_key="xgboost_hashed_ngram_http",
    )
    result_xgb["slo_verdicts"] = _audit_slos("xgboost+http", m_xgb, lat_xgb, slos) if slos else {}
    result_xgb["phase"] = phase
    existing["xgboost_hashed_ngram_http"] = result_xgb

    # ── Baseline C: Ablation — XGBoost without HTTP features ─────────────────
    if not args.no_ablation:
        log.info("\n" + "═" * 60)
        log.info("Baseline C: XGBoost + HashedNGram ONLY (ablation — no HTTP features)")
        log.info("═" * 60)
        result_abl, m_abl, lat_abl = _run_xgboost(
            X_train_ng, y_train, X_val_ng, y_val, X_test_ng, y_test,
            X_test_raw, test_classes, vectorizer,
            include_http=False, seed=cfg.project.seed,
            model_key="xgboost_hashed_ngram_ablation",
        )
        result_abl["slo_verdicts"] = _audit_slos("xgboost_ablation", m_abl, lat_abl, slos) if slos else {}
        result_abl["phase"] = phase
        existing["xgboost_hashed_ngram_ablation"] = result_abl

        # Log the feature-engineering lift for the paper
        lift_f1    = m_xgb["f1"]    - m_abl["f1"]
        lift_auc   = m_xgb["auc_pr"] - m_abl["auc_pr"]
        lift_fpr   = m_abl["fpr"]   - m_xgb["fpr"]
        log.info(f"\nHTTP feature lift: ΔF1={lift_f1:+.4f}  ΔAUC-PR={lift_auc:+.4f}  ΔFPR={lift_fpr:+.5f}")

    # ── Optional: LightGBM ────────────────────────────────────────────────────
    if args.also_lgbm:
        log.info("\n" + "═" * 60)
        log.info("Optional: LightGBM + HashedNGram + HTTP features")
        log.info("═" * 60)
        result_lgbm = _run_lightgbm(
            X_train_full, y_train, X_val_full, y_val, X_test_full, y_test,
            X_test_raw, test_classes, vectorizer,
            include_http=True, seed=cfg.project.seed,
        )
        if result_lgbm:
            result_lgbm["slo_verdicts"] = (
                _audit_slos("lightgbm", result_lgbm["overall"], result_lgbm["latency"], slos)
                if slos else {}
            )
            result_lgbm["phase"] = phase
            existing["lightgbm_hashed_ngram"] = result_lgbm

    # ── Write results ─────────────────────────────────────────────────────────
    baselines_path.write_text(json.dumps(existing, indent=2))
    log.info(f"\nAll classical ML baselines [{phase}] saved to {baselines_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name=f"03_classical_ml_baseline_{phase}"):
            mlflow.log_params({
                "phase":            phase,
                "also_lgbm":        args.also_lgbm,
                "no_ablation":      args.no_ablation,
                "hash_n_features":  HASH_N_FEATURES,
                "out_file":         args.out_file,
            })
            metrics: dict[str, float] = {}
            for key in ["aho_corasick_fast_match", "xgboost_hashed_ngram_http",
                        "xgboost_hashed_ngram_ablation", "lightgbm_hashed_ngram"]:
                if key not in existing:
                    continue
                e = existing[key]
                ov = e.get("overall", {})
                lat = e.get("latency", {})
                prefix = key.replace("_", ".")
                if ov.get("f1") is not None:
                    metrics[f"{prefix}.f1"]      = float(ov["f1"])
                if ov.get("fpr") is not None:
                    metrics[f"{prefix}.fpr"]     = float(ov["fpr"])
                if ov.get("auc_pr") is not None:
                    metrics[f"{prefix}.auc_pr"]  = float(ov["auc_pr"])
                if lat.get("throughput_rps") is not None:
                    metrics[f"{prefix}.rps"]     = float(lat["throughput_rps"])
            if "xgboost_hashed_ngram_http" in existing and "xgboost_hashed_ngram_ablation" in existing:
                ov_xgb = existing["xgboost_hashed_ngram_http"].get("overall", {})
                ov_abl = existing["xgboost_hashed_ngram_ablation"].get("overall", {})
                if ov_xgb.get("f1") and ov_abl.get("f1"):
                    metrics["lift_f1"]   = float(ov_xgb["f1"] - ov_abl["f1"])
                if ov_xgb.get("auc_pr") and ov_abl.get("auc_pr"):
                    metrics["lift_auc"]  = float(ov_xgb["auc_pr"] - ov_abl["auc_pr"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(baselines_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    # Summary table for the paper
    log.info("\n── Classical ML Summary (for paper Table) ──")
    log.info(f"{'Model':42s}  {'F1':>7}  {'FPR':>8}  {'AUC-PR':>7}  {'p99ms':>7}  {'RPS':>8}")
    log.info("─" * 82)
    for key in [
        "aho_corasick_fast_match",
        "xgboost_hashed_ngram_http",
        "xgboost_hashed_ngram_ablation",
        "lightgbm_hashed_ngram",
    ]:
        if key not in existing:
            continue
        e   = existing[key]
        ov  = e["overall"]
        lat = e.get("latency", {})
        log.info(
            f"{key:42s}  "
            f"F1={ov['f1']:.4f}  "
            f"FPR={ov['fpr']:.5f}  "
            f"AUC-PR={ov['auc_pr']:.4f}  "
            f"p99={lat.get('p99_ms', '?'):>5}ms  "
            f"RPS={lat.get('throughput_rps', '?'):>7}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2.2 — Classical ML and Fast-Match baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       default="config/pipeline.yaml")
    p.add_argument("--also-lgbm",    action="store_true",
                   help="Also run LightGBM (appended to output file, not in default run)")
    p.add_argument("--no-ablation",  action="store_true",
                   help="Skip the HTTP-feature ablation run (saves ~50%% training time)")
    p.add_argument("--split-dir",    default=None, metavar="DIR",
                   help="Directory containing train/val/test.parquet "
                        "(default: cfg.paths.data_splits)")
    p.add_argument("--out-file",     default="baselines.json", metavar="FILE",
                   help="Output filename within reports/metrics/ (default: baselines.json)")
    p.add_argument("--phase",        default="pre_aug", choices=["pre_aug", "post_aug"],
                   help="Pipeline phase tag written into each result entry "
                        "(pre_aug = before augmentation; post_aug = after augmentation). "
                        "Default: pre_aug")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())