# Stage 2 — Baselines

**Directory:** `stages/2_baselines/`  
**Make target:** `make baselines`  
**Outputs:** `data/splits/`, `reports/metrics/`

---

## Overview

Stage 2 establishes non-neural reference points against which the Transformer model is measured, and defines the SLOs that every subsequent evaluation stage checks against.

| Script | Role |
|---|---|
| `00_stratified_split.py` | Initial stratified train/val/test/adversarial/canary split |
| `01_define_slos.py` | Write SLO targets to `slos.json` |
| `02_modsecurity_crs.py` | OWASP CRS 3.3 heuristic baseline (PL1–PL4) |
| `03_classical_ml_baseline.py` | Aho-Corasick + HashedNGram + XGBoost baselines |

Stage 3 re-runs the stratified split on the augmented corpus (`07_stratified_split.py`), so the splits produced here are a *pre-augmentation* baseline. Both the pre- and post-augmentation splits are tagged in `split_stats.json` via the `_meta.phase` field.

---

## Scripts

### `00_stratified_split.py` — Initial stratified split

**Inputs:** `data/normalized/deduped.parquet`  
**Outputs:** `data/splits/{train,val,test,adversarial,canary}.parquet`, `reports/metrics/split_stats.json`

Produces five non-overlapping splits by stratifying on the composite key `label × attack_class`. This guarantees proportional representation of each attack type in every split, which is essential for per-class metrics in Stage 7 to be comparable across splits.

#### Sequential split algorithm

scikit-learn's `train_test_split` does not support carving more than two sets at once, so the five splits are produced in four sequential steps:

```
Full corpus
  └── canary (2%)      ← carved first
       └── adversarial (3% of remainder)
            └── test (10% of remainder)
                 └── val (15% of remainder)
                      └── train (remainder)
```

Canary is carved first because it is the smallest fraction and therefore the most likely to trigger the singleton-class fallback.

#### Stratification fallback

`_stratified_split()` wraps `train_test_split(..., stratify=valid_strat)`. Before stratifying, it null-masks any strata key that appears only once (can't be split). If sklearn raises `ValueError` despite this masking, the function falls back to a plain random split and sets `used_fallback=True`. Any split that triggered the fallback is recorded in `split_stats.json` → `_meta.fallback_splits` for audit.

#### Input selection

The script prefers `data/filtered/filtered.parquet` (output of Stage 3.4) if it exists, and falls back to `data/normalized/deduped.parquet`. This allows Stage 2 to run before Stage 3 has completed, which is the normal pipeline order.

---

### `01_define_slos.py` — SLO definition

**Inputs:** `config/pipeline.yaml`  
**Outputs:** `reports/metrics/slos.json`

Reads SLO parameters from config and writes them to a standalone JSON file that every evaluation script loads. Centralizing SLOs here means changing a threshold in config automatically propagates to all baseline and Stage 7 evaluations without touching their code.

The SLO document covers three dimensions:

| Dimension | Value | Notes |
|---|---|---|
| Inline WAF p99 latency | `cfg.slo.latency_inline_p99_ms` | batch=1, single request |
| Offline p99 latency | `cfg.slo.latency_offline_p99_ms` | batch=64, log-analysis mode |
| Minimum throughput | `cfg.slo.throughput_min_rps` | requests per second |
| Max false positive rate | `cfg.slo.max_false_positive_rate` | primary accuracy SLO |

Primary metric: `auc_pr`. The FPR ceiling is the hard constraint; AUC-PR is the optimization target.

---

### `02_modsecurity_crs.py` — ModSecurity CRS baseline

**Inputs:** `data/splits/test.parquet`, `reports/metrics/slos.json`  
**Outputs:** `reports/metrics/modsecurity_results.json`, `reports/metrics/baselines.json`

Implements OWASP CRS 3.3 as a compiled-regex anomaly scoring engine, without a real ModSecurity installation.

#### Rule structure

Each `CrsRule` carries:
- `attack_class` — canonical class label for per-class breakdown
- `severity` — score contribution per match (CRITICAL=5, ERROR=4, WARNING=3, NOTICE=2)
- `min_pl` — minimum paranoia level at which this rule is active

19 rules total, grouped into four paranoia levels:

| Level | Description |
|---|---|
| PL1 | Core rules — high confidence, near-zero FP. Production default (7 rules). |
| PL2 | Moderately aggressive — stricter SQL/XSS checks, more tautology detection (4 rules). |
| PL3 | Strict — stored proc enumeration, attribute-injected JS, header injection (4 rules). |
| PL4 | Maximum paranoia — bare SQL keywords, any inline event handler, Unix path fragments (3 rules). |

#### Anomaly scoring — `score_request()`

All rules active at the configured paranoia level are evaluated against `raw.lower()`. Each matching rule adds its severity score to a running total. A request is flagged when the total meets or exceeds `inbound_anomaly_score_threshold` (default 5, matching CRS defaults). This replaces the original binary match/no-match approach and mirrors the real CRS `SecInboundAnomalyScoreThreshold` mechanism.

The accumulated score is passed to `compute_metrics()` as a soft probability proxy, enabling AUC-ROC and AUC-PR computation alongside the binary F1/FPR metrics.

#### SLO audit — `_audit_slos()`

After each paranoia level is evaluated, its results are checked against the three SLO dimensions. Verdicts are independent: an accuracy PASS with a latency FAIL is the expected outcome for CRS (regex throughput easily satisfies the latency SLO; accuracy does not meet the FPR floor). Both verdicts are recorded separately in `baselines.json`.

#### Expected research outcome

```
PL1  : Latency PASS  /  Accuracy FAIL  (low recall)
PL2  : Latency PASS  /  Accuracy FAIL  (better recall, higher FP)
PL3-4: Latency PASS  /  Accuracy FAIL  (FP rate becomes untenable)
```

This establishes that signature-based approaches have a hard ceiling, motivating the ML models.

#### CLI flags

| Flag | Effect |
|---|---|
| `--paranoia-level {1,2,3,4}` | Evaluate a single PL instead of all four |
| `--threshold INT` | Override anomaly score threshold |
| `--audit-log PATH` | Parse a real ModSecurity audit log instead of using the heuristic engine |

---

### `03_classical_ml_baseline.py` — Classical ML baselines

**Inputs:** `data/splits/{train,val,test}.parquet`, `reports/metrics/slos.json`  
**Outputs:** `reports/metrics/baselines.json`

Three qualitatively distinct baselines that together establish the research narrative:

```
Fast-Match  → Maximum throughput, minimum accuracy  (regex upper bound)
XGBoost+B   → Best classical accuracy              (primary ML baseline)
XGBoost+A   → XGBoost without HTTP features        (feature ablation)
```

LightGBM is available via `--also-lgbm` but excluded from the default run because it and XGBoost are both gradient-boosted trees over the same feature space, adding wall-clock cost without distinct qualitative insight.

#### Baseline A — Aho-Corasick fast-match (`_run_fast_match()`)

A compiled automaton over `_AC_SIGNATURES` (60 high-confidence attack signatures). Uses the `pyahocorasick` package when available; falls back to a plain Python `any(sig in text for sig in sigs)` loop that produces identical predictions at ~10× lower throughput.

Latency is measured per-request over a 500-sample probe (the automaton is per-request, not batched). Results are expressed in microseconds — expected p99 in the single-digit µs range, well under the inline WAF SLO.

The intentionally compact signature list understates accuracy and overstates throughput, which is the conservative bound the research paper wants.

#### Baseline B — HashedNGram + XGBoost + HTTP features (`_run_xgboost()`)

**Vectorizer:** `HashingVectorizer(analyzer="char_wb", ngram_range=(1,3), n_features=2²⁰, norm="l2", alternate_sign=False)`. Using `HashingVectorizer` instead of TF-IDF simulates production memory constraints (no vocabulary dict stored; ~4 MB fixed RAM). `alternate_sign=False` keeps values non-negative for XGBoost.

**HTTP field features — `_http_features()`:** A 7-dimensional dense vector appended to the sparse n-gram matrix:

| Feature | Signal |
|---|---|
| `has_sqli_in_query` | Single-quote or UNION in query string value |
| `has_sqli_in_body` | Tautology pattern in POST body |
| `has_xss_in_ua` | Script tag in User-Agent |
| `has_path_traversal` | `../` repeated ≥2× in request line |
| `has_encoded_payload` | Percent-encoded special chars repeated ≥2× |
| `has_long_value` | Normalized request length (continuous 0–1) |
| `has_many_params` | Boolean: more than 10 query params |

The HTTP features capture context that character n-grams cannot: a single-quote in the User-Agent is almost always benign; the same character in a query-string value is a SQLi signal.

**Training:** `XGBClassifier` with `scale_pos_weight = n_neg / n_pos` for imbalance. GPU acceleration auto-detected via `_xgb_device_kwargs()`: XGBoost ≥ 2.0 uses `device=cuda`; older versions use `tree_method=gpu_hist`. Early stopping at 50 rounds on val AUC-PR.

**Latency benchmark:** Re-vectorizes a fixed 64-sample probe batch on every run (10 warmup, 50 timed) to include vectorizer cost, which is incurred on every production request. Reports p50/p95/p99/throughput.

#### Baseline C — XGBoost without HTTP features (ablation)

Identical to Baseline B but trained on the bare n-gram matrix (no `_http_features()` columns). The feature-engineering lift is logged and MLflow-tracked:

```
HTTP feature lift: ΔF1=+0.0312  ΔAUC-PR=+0.0481  ΔFPR=-0.00412
```

#### CLI flags

| Flag | Effect |
|---|---|
| `--also-lgbm` | Add LightGBM run (appended to `baselines.json`) |
| `--no-ablation` | Skip Baseline C (saves ~50% training time) |
| `--split-dir DIR` | Override input directory (default: `cfg.paths.data_splits`) |
| `--out-file FILE` | Output filename within `reports/metrics/` (default: `baselines.json`) |
| `--phase {pre_aug,post_aug}` | Tag written into each result entry |

The `--phase` flag is used by Stage 3 which re-runs the baselines on the augmented corpus with `--phase post_aug`, writing to the same `baselines.json`. Both entries coexist under different keys.

---

## Data flow

```
data/normalized/deduped.parquet
        │
        ▼
00_stratified_split.py
        │
        ├──▶ data/splits/train.parquet
        ├──▶ data/splits/val.parquet
        ├──▶ data/splits/test.parquet
        ├──▶ data/splits/adversarial.parquet
        └──▶ data/splits/canary.parquet
                    │
config/pipeline.yaml ──▶ 01_define_slos.py ──▶ reports/metrics/slos.json
                                                        │
data/splits/test.parquet                                │
        │                                               │
        ├──▶ 02_modsecurity_crs.py ──────────────▶ baselines.json (modsecurity_crs_pl{1,2,3,4})
        │                                               │
        └──▶ 03_classical_ml_baseline.py ──────────▶ baselines.json (aho_corasick, xgboost_*)
```

---

## Configuration

Relevant keys in `config/pipeline.yaml`:

```yaml
slo:
  latency_inline_p99_ms:  5      # Single-request inline WAF SLO
  latency_offline_p99_ms: 50     # Batch=64 offline SLO
  throughput_min_rps:     1000   # Minimum requests per second
  max_false_positive_rate: 0.001 # Hard FPR ceiling
  crs_anomaly_threshold:  5      # ModSecurity score threshold

data:
  split:
    train:       0.70
    val:         0.15
    test:        0.10
    adversarial: 0.03
    canary:      0.02

project:
  seed: 42
```

---

## Running the stage

```bash
# Full stage
make baselines

# Individual scripts
python stages/2_baselines/00_stratified_split.py --config config/pipeline.yaml
python stages/2_baselines/01_define_slos.py      --config config/pipeline.yaml
python stages/2_baselines/02_modsecurity_crs.py  --config config/pipeline.yaml
python stages/2_baselines/03_classical_ml_baseline.py --config config/pipeline.yaml

# Single paranoia level only
python stages/2_baselines/02_modsecurity_crs.py --paranoia-level 1

# Classical baselines with LightGBM comparison
python stages/2_baselines/03_classical_ml_baseline.py --also-lgbm

# Skip HTTP-feature ablation (faster)
python stages/2_baselines/03_classical_ml_baseline.py --no-ablation
```

---

## What to check after running

| Check | Where |
|---|---|
| Split sizes and class balance | `reports/metrics/split_stats.json` → per-split `imbalance_ratio` |
| Fallback splits | `reports/metrics/split_stats.json` → `_meta.fallback_splits` (should be `[]`) |
| SLO targets | `reports/metrics/slos.json` |
| CRS recall vs FPR by paranoia level | `reports/metrics/baselines.json` → `modsecurity_crs_pl{1..4}.overall` |
| CRS SLO verdicts | `reports/metrics/baselines.json` → `modsecurity_crs_pl1.slo_verdicts` |
| XGBoost vs ablation lift | `reports/metrics/baselines.json` → compare `xgboost_hashed_ngram_http` vs `xgboost_hashed_ngram_ablation` |
| Latency benchmark | `reports/metrics/baselines.json` → `latency.p99_ms` per model |
| MLflow | `make ui` → experiments `00_stratified_split`, `02_modsecurity_crs`, `03_classical_ml_baseline_pre_aug` |

A stratification fallback on a non-canary split is worth investigating: it means some class has only one sample in the corpus and the split is not genuinely stratified for that class. Stage 3 augmentation targets exactly these gaps.
