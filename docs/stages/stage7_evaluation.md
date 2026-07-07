# Stage 7 — Evaluation

> **📖 Docs:** [Index](../README.md) · [User Guide](../USER_GUIDE.md) · [Architecture](../ARCHITECTURE.md) · [API](../API.md) · [All Stages](stages.md) · [Model Card](../MODEL_CARD.md)

**Directory:** `stages/7_evaluation/`  
**Make targets:** `make eval_detection`, `make eval_latency`, `make eval_adversarial`, `make eval_ablation`, `make eval_interp`, `make compare_all`, `make report`  
**Outputs:** `reports/7_evaluation/{metrics,latency}/*.json`, `reports/7_evaluation/15_final_evaluation_report.json`, `reports/7_evaluation/metrics/14_master_comparison_table.{csv,json}`, `reports/7_evaluation/metrics/16_deployment_recommendation.json`

---

## Purpose

Stage 7 is the full evaluation suite that produces all results referenced in the paper. It covers four concerns:

1. **Detection efficacy** — how well each model detects attacks vs. the SLOs
2. **Latency & memory** — whether models meet inline (<5 ms p99) and offline (<50 ms p99) SLOs
3. **Adversarial robustness** — whether detection holds under encoding mutations, chained obfuscation, and novel attack families
4. **Ablations & interpretability** — what each design decision (tokenizer, augmentation, model size, label smoothing) contributes

Every script is idempotent: it skips its output if already present unless `--force` is passed. All metrics are also logged to MLflow (`make ui` → `localhost:5000`).

---

## Script overview

| Script | Purpose | Key output |
|---|---|---|
| `01_detection_metrics.py` | Full metric suite on test split for all models | `reports/7_evaluation/metrics/01_detection_results.json` |
| `02_latency_bench.py` | p50/p95/p99/p99.9 latency + throughput; SLO verdict | `reports/7_evaluation/latency/02_latency_summary.json` |
| `03_memory_footprint.py` | VRAM, RAM, and checkpoint size for all models | `reports/7_evaluation/metrics/03_memory_footprint.json` |
| `04_evasion_payloads.py` | Detection rate against all TAMPER_REGISTRY transforms | `reports/7_evaluation/metrics/04_adversarial_summary.json` |
| `05_obfuscation_robustness.py` | Chained multi-tamper evasion attempt | `reports/7_evaluation/metrics/05_obfuscation_robustness.json` |
| `06_novel_attack_generalization.py` | Grammar-fuzzed novel attack families with 95% CIs | `reports/7_evaluation/metrics/06_novel_attack_generalization.json` |
| `07_tokenizer_ablation.py` | Track A vs Track B with transformer-corrected proxy estimate | `reports/7_evaluation/metrics/07_tokenizer_ablation.json` |
| `08_augmentation_ablation.py` | With vs. without synthetic augmentation | `reports/7_evaluation/metrics/08_augmentation_ablation.json` |
| `09_model_size_scaling.py` | AUC-PR vs. parameter count (scaling curve) | `reports/7_evaluation/metrics/09_model_size_scaling.json` |
| `10_label_smoothing_ablation.py` | Effect of label smoothing on calibration and F1 | `reports/7_evaluation/metrics/10_label_smoothing_ablation.json` |
| `11_attention_visualization.py` | Attention heatmaps per attack class; head specialisation | `reports/7_evaluation/metrics/11_attention_visualization.json` |
| `12_shap_analysis.py` | KernelSHAP token importance on student model | `reports/7_evaluation/metrics/12_shap_analysis.json` |
| `13_error_analysis.py` | FP/FN characterisation by attack class | `reports/7_evaluation/metrics/13_error_analysis.json` |
| `14_comparison_table.py` | Master table: all models × all metrics (CSV + JSON) | `reports/7_evaluation/metrics/14_master_comparison_table.{csv,json}` |
| `15_generate_report.py` | Consolidated JSON report from all eval outputs | `reports/7_evaluation/15_final_evaluation_report.json` |
| `16_deployment_recommendation.py` | Weighted scoring matrix → deployment decision | `reports/7_evaluation/metrics/16_deployment_recommendation.json` |
| `17_push_to_hub.py` | Push all artifacts to HuggingFace Hub | — |

---

## Script details

### 7.1 — `01_detection_metrics.py`

Runs inference on the test split for all available models and computes the core detection metrics at the primary operating point (FPR = 0.001).

**Models evaluated:**
- Track B 99M (`best_99m.pt`) — the teacher
- Student INT8 (`best_student.pt`) — the distilled model
- XGBoost baseline — read from `reports/2_baselines/metrics/03_baselines.json` (produced by Stage 2)
- ModSecurity CRS — read from `reports/2_baselines/metrics/03_baselines.json`

**Metrics per model and per attack class:**
- F1, Precision, Recall, FPR, FNR, AUC-ROC, AUC-PR

**FP diagnostic categorisation** (two buckets):
- *Safe-but-Malformed*: false positives with Shannon entropy < 5.5 bits/byte (low-entropy text that pattern-matches attack signatures)
- *High-Entropy*: false positives with Shannon entropy ≥ 5.5 bits/byte (encrypted payloads, compressed data, base64 blobs)

**FP cluster analysis**: K-Means (k=5) over the CLS token embedding of false positive samples, with PCA projection to 2D. Identifies systematic misclassification patterns that can be fed back into Stage 3 augmentation.

---

### 7.2 — `02_latency_bench.py`

Benchmarks all models at batch sizes [1, 8, 32, 64, 256] on both GPU and CPU. Reports p50, p95, p99, and p99.9 latency plus throughput (RPS). Also measures PCIe host→device transfer overhead separately per batch size.

Primary SLO checks (from `config/pipeline.yaml`):
- **p99 at bs=1** ≤ 5 ms (inline appliance)
- **p99 at bs=64** ≤ 50 ms (offline batch)

The p99.9 tail and per-run jitter (standard deviation across warmup-excluded iterations) are also recorded for QoS analysis but do not gate the SLO verdict.

---

### 7.3 — `03_memory_footprint.py`

Reports peak VRAM during inference, RSS RAM, activation memory, and checkpoint file sizes (`.pt`, `.onnx`) for all models. Useful for sizing deployment hardware.

---

### 7.4 — `04_evasion_payloads.py`

Runs the eight tamper functions from `TAMPER_REGISTRY` against each model on the adversarial holdout split. Also accepts external evasion wordlists from `config`.

| Tamper | Technique |
|---|---|
| `tamper_space2comment` | `/**/` comment substitution |
| `tamper_randomcase` | Random character casing |
| `tamper_url_encode` | Single URL encoding |
| `tamper_double_url_encode` | Double URL encoding |
| `tamper_hex_encode` | Hex literal encoding |
| `tamper_between_comments` | SQL comment insertion |
| `tamper_base64_param` | Base64 wrapping |
| `tamper_unicode_escape` | Unicode escape sequences |

Reports evasion rate (fraction of mutated attacks that evade detection) per tamper and per model.

---

### 7.5 — `05_obfuscation_robustness.py`

Tests robustness to chained multi-tamper sequences (e.g., URL-encode → randomcase → comment insertion applied in sequence). A single tamper may be caught by vocabulary overlap; chaining degrades the token distribution more aggressively. Reports per-chain evasion rate.

---

### 7.6 — `06_novel_attack_generalization.py`

Evaluates generalisation to attack families that were **not present during training**. Instead of fixed sample strings, a grammar-based fuzzer generates 500–1000 syntactic variations per attack class using lightweight production rules, making detection rates statistically stable.

Reports detection rate with **95% Wilson score confidence intervals** over the generated variations. Researchers can add new attack grammars without modifying core evaluation logic.

---

### 7.7 — `07_tokenizer_ablation.py`

Compares Track A (augmented BERT vocab) vs Track B (custom BPE) on OOV rate, token fertility, and downstream detection performance.

Because a full Transformer re-training for each tokenizer is expensive, this script uses a two-stage **Validation Anchor** method:

1. Train a fast LR + TF-IDF probe for both Track A and Track B.
2. Run a truncated Transformer fine-tune (5 epochs, 20% of data by default) for both tracks.
3. Compute a per-metric **scaling factor**: `transformer_delta / probe_delta`.
4. Apply the scaling factor to the probe results to produce a *transformer-corrected* performance estimate.

Both the raw probe deltas and the corrected estimates are reported so researchers can assess uncertainty in the proxy.

---

### 7.8 — `08_augmentation_ablation.py`

Trains an LR + TF-IDF probe on the full augmented corpus and on the pre-augmentation corpus, then reports the metric delta attributable to synthetic augmentation. The probe is fast enough to run without a GPU and isolates the augmentation variable without requiring full Transformer retraining.

---

### 7.9 — `09_model_size_scaling.py`

Reads `detection_results.json` (from 7.1) and `baselines.json` and reconstructs the AUC-PR vs. parameter count scaling curve. Does not train new models; instead reports metrics already logged for checkpoints of different sizes.

---

### 7.10 — `10_label_smoothing_ablation.py`

Compares calibration (ECE) and F1 with vs. without label smoothing using a probe model. Reports ECE, F1, and the fraction of predictions that fall in each confidence decile.

---

### 7.11 — `11_attention_visualization.py`

For each attack class, extracts attention weights from the final encoder layer of the teacher model over representative malicious samples. Identifies which token positions are attended to and whether individual heads specialise (e.g., one head focuses on SQL keywords, another on URL structure).

Saves heatmap data as JSON for offline rendering with any plotting tool (matplotlib, plotly, seaborn).

---

### 7.12 — `12_shap_analysis.py`

Runs KernelSHAP on the **student model** (small enough for CPU-based SHAP). Uses 50 background benign samples and explains 5 malicious samples. Reports per-token SHAP values showing which input positions drove the malicious classification.

Requires `pip install shap`; skips gracefully with an error log if not installed.

---

### 7.13 — `13_error_analysis.py`

Characterises the false positive and false negative populations by attack class. Reports:
- Most common FP attack classes (attacks misclassified as benign)
- Most common FN patterns (benign traffic misclassified as malicious)
- Confusion matrix breakdown per attack family

---

### 7.14 — `14_comparison_table.py`

Aggregates all prior eval JSON outputs into a single master comparison table. Pulls from:

| Source file | Fields extracted |
|---|---|
| `detection_results.json` | F1, AUC-PR, FPR, Recall |
| `latency_summary.json` | p99_ms, p99.9_ms, jitter, RPS at bs=1 |
| `adversarial_summary.json` | mean evasion rate per model |
| `novel_attack_generalization.json` | mean detection rate across novel classes |
| `memory_footprint.json` | checkpoint size MB, param count |

Writes `master_comparison_table.csv` (human-readable) and `master_comparison_table.json` (consumed by `16_deployment_recommendation.py`).

---

### 7.15 — `15_generate_report.py`

Consolidates all Stage 7 JSON outputs into a single structured `final_evaluation_report.json`. Includes every subsection: detection, latency, memory, adversarial, ablations, interpretability summaries, error analysis, FP clustering, and the deployment recommendation if `16` has run.

---

### 7.16 — `16_deployment_recommendation.py`

Computes a weighted deployment score for each model:

```
score(model) =
    0.5 × Normalized(AUC-PR)
  + 0.3 × Normalized(latency_score)       # 1.0 if p99 ≤ SLO, degrades linearly above
  + 0.2 × Normalized(1 − mean_evasion_rate)
```

Reads `master_comparison_table.json` (from 7.14) and outputs a ranked recommendation with the rationale for the top-scored model. Exits with code 1 if no model passes all SLOs, so CI/CD can gate a deployment pipeline on this step.

---

### 7.17 — `17_push_to_hub.py`

Convenience wrapper that pushes all existing artifacts to HuggingFace Hub in one shot. Individual stage scripts can also push incrementally via `--save / --revision`; this script handles the bulk upload.

```bash
export HF_ORG=your-org-name
export HF_TOKEN=hf_...          # or: huggingface-cli login

python stages/7_evaluation/17_push_to_hub.py            # push everything
python stages/7_evaluation/17_push_to_hub.py --dry-run  # preview only
python stages/7_evaluation/17_push_to_hub.py --revision v1.0-aug
python stages/7_evaluation/17_push_to_hub.py --only models   # or: datasets
```

---

## Make targets

| Target | What it runs |
|---|---|
| `make eval_detection` | 7.1 |
| `make eval_latency` | 7.2, 7.3 |
| `make eval_adversarial` | 7.4, 7.5, 7.6 |
| `make eval_ablation` | 7.7, 7.8, 7.9, 7.10 |
| `make eval_interp` | 7.11, 7.12, 7.13 |
| `make compare_all` | All of the above + baselines re-run (post-aug) + 7.14 |
| `make report` | 7.15 |
| `make eval_only` | `eval_detection` + `eval_latency` + `eval_adversarial` + `eval_ablation` + `eval_interp` + `compare_all` |
| `make push_hub` | 7.17 |
| `make push_hub_dry` | 7.17 `--dry-run` |

---

## Data flow

```
Stage 6 → models/student/best_student.pt
           models/student/student.onnx
           models/track_b/99m/best_99m.pt
Stage 3 → data/splits/test.parquet
           data/splits/adversarial.parquet
Stage 4 → tokenizers/track_b/
Stage 2 → reports/2_baselines/metrics/03_baselines.json
           reports/slos.json

  01_detection_metrics        →  reports/7_evaluation/metrics/01_detection_results.json
  02_latency_bench            →  reports/7_evaluation/latency/02_latency_summary.json
  03_memory_footprint         →  reports/7_evaluation/metrics/03_memory_footprint.json
  04_evasion_payloads         →  reports/7_evaluation/metrics/04_adversarial_summary.json
  05_obfuscation_robustness   →  reports/7_evaluation/metrics/05_obfuscation_robustness.json
  06_novel_attack_*           →  reports/7_evaluation/metrics/06_novel_attack_generalization.json
  07_tokenizer_ablation       →  reports/7_evaluation/metrics/07_tokenizer_ablation.json
  08_augmentation_ablation    →  reports/7_evaluation/metrics/08_augmentation_ablation.json
  09_model_size_scaling       →  reports/7_evaluation/metrics/09_model_size_scaling.json
  10_label_smoothing_ablation →  reports/7_evaluation/metrics/10_label_smoothing_ablation.json
  11_attention_visualization  →  reports/7_evaluation/metrics/11_attention_visualization.json
  12_shap_analysis            →  reports/7_evaluation/metrics/12_shap_analysis.json
  13_error_analysis           →  reports/7_evaluation/metrics/13_error_analysis.json
  14_comparison_table         →  reports/7_evaluation/metrics/14_master_comparison_table.{csv,json}
  15_generate_report          →  reports/7_evaluation/15_final_evaluation_report.json
  16_deployment_recommendation → reports/7_evaluation/metrics/16_deployment_recommendation.json
```

---

## Running

```bash
# Full evaluation suite (recommended)
make compare_all
make report

# Individual groups
make eval_detection
make eval_latency
make eval_adversarial
make eval_ablation
make eval_interp

# Force re-run of a specific script
python stages/7_evaluation/01_detection_metrics.py --config config/pipeline.yaml --force

# Push artifacts to HuggingFace
make push_hub
make push_hub_dry                               # preview only
make push_hub HF_REVISION=v1.0 HF_ONLY=models  # tag a release
```

### Optional dependencies

| Feature | Required package |
|---|---|
| SHAP analysis (7.12) | `shap` — `pip install shap` |
| HuggingFace push (7.17) | `huggingface_hub` — included in `make setup MODE=full` |

---

[◀ Stage 6 — Distillation & Compression](stage6_distillation_and_compression.md) · [All Stages ▲](stages.md) · _(last stage)_ ▶
