# Pipeline Stages Reference

The pipeline is composed of seven sequential stages under `stages/`. Each stage is a numbered directory containing standalone Python scripts. Stages must be run in order; later stages depend on artefacts produced by earlier ones.

Supported stages can push their outputs to HuggingFace Hub immediately after completing — see [docs/huggingface.md](huggingface.md).

---

## Re-running Stages

Every entry-point script in the pipeline is **idempotent by default** and enforces **prerequisite checks** before doing any work.

### Prerequisite guards

At startup each script verifies that its required inputs exist. If a prerequisite is missing the script exits immediately with a clear error rather than failing midway through:

```
ERROR  Required input missing: data/normalized/deduped.parquet  →  run: make data_collect
ERROR  Required input missing: reports/metrics/taxonomy_inventory.json  →  run: make data_analyze
```

### Skip-if-done guards

If a script's primary output file already exists and is non-empty, the script skips all work and exits cleanly:

```
INFO   Corpus report already exists (42,317 bytes) — skipping. Pass --force to re-run.
```

This makes `make data_collect` safe to re-run at any time — only scripts whose outputs are missing (or stale after a `--force`) will do any real work.

### `--force` flag

Every script accepts `--force` to bypass the skip-if-done guard and re-run unconditionally. Prerequisite checks are always enforced even with `--force`.

```bash
# Re-generate corpus report even if it already exists
python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py --force

# Re-run the quality gate
python stages/3_data_augmentation/04_quality_gate.py --force
```

### Script-specific override flags

Two scripts have additional, more granular flags alongside `--force`:

| Script | Flag | Scope |
|---|---|---|
| `01_acquire_and_normalize.py` | `--force` | Re-download **and** re-ingest all datasets |
| `01_acquire_and_normalize.py` | `--force-ingest` | Re-ingest from existing CSVs without re-downloading |
| `02_cross_dataset_dedup.py` | `--force-dedup` | Re-run deduplication even if `deduped.parquet` exists |

### Make-level re-runs

All make targets pass through to the underlying scripts, so `--force` can be threaded through via `ARGS`:

```bash
# Re-run just the corpus report
python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py --config config/pipeline.yaml --force

# Re-run a full stage (each script checks its own outputs independently)
make data_augment_synthesis  # skips scripts whose outputs exist
```

### Dependency chain

The full prerequisite chain is:

```
01_acquire_and_normalize  →  data/normalized/all_datasets.parquet
02_cross_dataset_dedup    →  data/normalized/deduped.parquet
03_generate_corpus_report →  reports/metrics/taxonomy_inventory.json
                             reports/corpus_report.json
00_stratified_split       →  data/splits/{train,val,test,adversarial,canary}.parquet
01_attack_synthesis       →  data/augmented/synthesis/synthesized_attacks.parquet
02_request_framing        →  data/augmented/framed/framed_records.parquet
04_quality_gate           →  data/augmented/filtered/filtered_records.parquet
07_stratified_split       →  data/splits/ (augmented, final)
03_train_custom_bpe       →  tokenizers/track_b/
03_track_b_99m            →  models/track_b/99m/best_99m.pt
02_distill_train          →  models/student/best_student.pt
05_export_and_bench       →  models/student/student.onnx
```

If a prerequisite is absent, the dependent script will print the exact make target to run.

---

## Stage 1 — Data Acquisition & Curation

**Directory:** `stages/1_data_acquisition_and_curation/`  
**Make target:** `make data_collect`

Downloads public HTTP datasets, Normalizes them to the canonical `HttpRecord` format, deduplicates across sources, and produces a corpus report.

### Scripts

#### `01_acquire_and_normalize.py`

Single-pass download → Normalize → Parquet writer. For each configured dataset:

1. Checks for a local copy; downloads via Kagglehub with SHA-256 verification if absent
2. Adapts the source schema to `HttpRecord` using per-dataset converters (`CLASS_ALIASES` canonicalisation)
3. Streams records through Pydantic validation and writes to Parquet in batches of 10 000

Datasets handled: CSIC 2010, SR-BH 2020, ECML/PKDD 2007, and additional sources declared in `config/pipeline.yaml`. Output lands in `data/normalized/`.

#### `02_cross_dataset_dedup.py`

Two-pass deduplication across all Normalized Parquet files:

- **Pass 1** — exact SHA-256 hash of the `raw` field; drops byte-identical records
- **Pass 2** — MinHash LSH at configurable Jaccard threshold (default 0.85); drops near-duplicates

Writes deduplicated output back to `data/normalized/` and logs removal counts to MLflow.

#### `03_generate_corpus_report.py`

Single-pass analysis that replaces five separate reporting scripts:

| Report section | Content |
|---|---|
| Dataset analysis | Overall counts, label imbalance ratio, HTTP method distribution |
| Taxonomy inventory | Per-class coverage; flags missing or low-count classes |
| Taxonomy coverage | Class × source coverage matrix |
| Length distribution | Character and token length percentiles |
| Datasheet stub | Gebru et al. metadata for the assembled corpus |

Output: `reports/corpus_report.json` and `reports/corpus_report.html`.

---

## Stage 2 — Baselines

**Directory:** `stages/2_baselines/`  
**Make target:** `make baselines`

Establishes non-neural reference points against which the Transformer model is measured.

### Scripts

#### `00_stratified_split.py`

Stratified train / val / test / adversarial / canary split, stratified by `(label × attack_class)`. Falls back to random split for rare classes that cannot satisfy the stratification constraint. Writes one Parquet file per split to `data/splits/`.

#### `01_define_slos.py`

Writes the SLO targets to `reports/slos.json`:

| SLO | Value |
|---|---|
| p99 latency (inline) | < 5 ms |
| p99 latency (offline) | < 50 ms |
| Maximum FPR | 0.001 |
| Minimum throughput | configurable RPS |

#### `02_modsecurity_crs.py`

OWASP ModSecurity CRS 3.3 heuristic baseline with four Paranoia Levels (PL1–PL4):

- 19 CRS rules with severity-weighted anomaly scoring
- **PL1** — high-confidence rules only; near-zero false positives (production default)
- **PL2–PL4** — progressively aggressive; trade accuracy for coverage
- Produces per-class metrics, SLO PASS/FAIL verdicts, and a latency estimate

#### `03_classical_ml_baseline.py`

Three classical baselines in a single script:

| Variant | Description |
|---|---|
| A — Aho-Corasick | Pure string matching; throughput upper bound |
| B — HashedNGram + XGBoost | Primary baseline; 2²⁰ hash buckets, GPU XGBoost, early stopping on AUC-PR |
| C — No-HTTP-features ablation | Same as B but without structural HTTP field features; quantifies field-awareness lift |

HTTP field feature vector (7 dims): SQLi patterns in query string and body, XSS in User-Agent, path traversal, encoding indicators, request length, parameter count. Optional LightGBM comparison run when the package is present.

---

## Stage 3 — Data Augmentation

**Directory:** `stages/3_data_augmentation/`  
**Make target:** `make data_augment_all`

Synthesises additional training samples to fill taxonomy gaps and balance class distributions.

### Scripts

#### `01_attack_synthesis.py`

Unified attack payload generation orchestrated by `AugmentationGovernor`:

1. Reads `taxonomy_inventory.json` to compute per-class sample gaps
2. Dispatches to a generator registry based on gap size and class type

**Generator types:**

| Generator | Mechanism |
|---|---|
| `GrammarGenerator` | Context-free templates for SQLi, XSS, LFI, SSRF, CMDi with prefixes / payloads / suffixes / params / endpoints |
| `MutatorGenerator` | 8 encoding transforms applied to existing payloads |
| `TamperGenerator` | 7 SQLMap-style tamper rules (comment injection, casing, URL/hex/base64/unicode encoding) |
| `LocalLLMGenerator` | Local LLM via Ollama API for out-of-grammar payloads |

Mutation chains apply transforms probabilistically (Grammar → Tamper → Encode). ThreadPoolExecutor parallelises generation per class. All outputs are validated through `HttpRecord`.

#### `02_request_framing.py`

Wraps bare payloads in realistic HTTP envelopes:

- **`HttpMetadataDistribution`** (singleton) — shared header / path / User-Agent frequency tables used for both attack and benign framing to prevent synthetic fingerprinting
- **Attack framing** — injects payload into GET and POST envelopes with realistic metadata
- **Benign REST generator** — programmatic endpoints covering auth, search, pagination, and CRUD patterns
- **Cloud LLM benign** — Anthropic / Google APIs generate edge-case benign samples (SQL words in natural English, HTML-like bodies, relative paths)

#### `03_benign_enrichment.py`

Aligns the benign generator's metadata distributions with real traffic:

- Extracts header and path frequencies from PCAP files (dpkt with scapy fallback)
- Learned distributions parameterise the REST generator in `02_request_framing.py`
- Falls back to internal defaults when no PCAP is provided

#### `04_quality_gate.py`

Four-pass filtering pipeline applied to all synthetic samples:

| Pass | Mode | Check |
|---|---|---|
| 1 | Parallel | HTTP format validation: method allowlist, length bounds |
| 2 | Parallel | Tokenizer UNK-rate: reject samples above threshold |
| 3 | Sequential | Semantic dedup via MinHash LSH |
| 4 | Sequential | CRS-aligned label consistency check |

Additionally runs a **leakage guard**: removes synthetic records with Jaccard similarity > 0.70 to any record in the test or canary splits. Conflicting benign edge-cases are quarantined rather than deleted.

#### `05_augmentation_probe.py`

Diagnostic script that trains a lightweight probe model on augmented vs. original data to measure whether synthetic samples improve generalisation.

#### `06_taxonomy_inventory.py`

Recomputes the taxonomy inventory after augmentation; verifies that per-class gaps have been closed.

#### `07_stratified_split.py`

Re-runs stratified split on the augmented corpus (same logic as `2_baselines/00_stratified_split.py`) to produce the final `data/splits/` files used by all downstream training stages.

---

## Stage 4 — Tokenization

**Directory:** `stages/4_tokenization/`  
**Make targets:** `make tokenize_b` (Track B); Track A scripts run individually

Compares two tokenization strategies and produces artefacts for training.

### Scripts

| Script | Purpose |
|---|---|
| `01_augment_pretrained_vocab.py` | **Track A** — add HTTP-specific tokens to BERT-base-uncased; runs shadow analysis to verify tokens are treated as atomic units; saves augmented tokenizer to `tokenizers/track_a/` |
| `02_measure_oov_track_a.py` | OOV rate and token fertility on the corpus for Track A |
| `03_train_custom_bpe.py` | **Track B** — train byte-level BPE from scratch on HTTP corpus; saves tokenizer to `tokenizers/track_b/` |
| `04_measure_oov_track_b.py` | OOV rate and token fertility on the corpus for Track B |
| `05_compare_tokenizers.py` | Side-by-side comparison: OOV rate, fertility, vocabulary Jaccard overlap, per-attack-class OOV breakdown |
| `tokenizer_eval.py` | Extended evaluation utilities; token shadowing analysis |

**Track A** augments an existing pretrained tokenizer — faster to set up, benefits from pretraining signal.  
**Track B** trains BPE entirely from the HTTP corpus — no pretrained weights, but vocabulary is tuned exactly to HTTP structure.

---

## Stage 5 — Teacher Training

**Directory:** `stages/5_teacher_training/`  
**Make target:** `make train_b_99m`

Trains three teacher model configurations.

### Scripts

#### `01_track_a_large.py`

Fine-tunes DeBERTa-v3-base (337M params) on the WAF dataset using the Track A tokenizer. Logs to MLflow under experiment `track_a_large`.

#### `02_track_a_small.py`

Fine-tunes a smaller DeBERTa variant for use as a compression target. Produces a lighter Track A teacher for distillation experiments.

#### `03_track_b_99m.py`

Trains the custom 99M-parameter `WafEncoder` from scratch with the Track B tokenizer. Includes curriculum learning: early epochs use shorter sequences; full-length training begins after a warmup period.

#### `03b_distill_track_b.py`

Distils the Track A teacher into a Track B student, transferring knowledge across tokenization tracks. Acts as a cross-track bridge when the Track A teacher outperforms the from-scratch Track B teacher.

#### Shared utilities

| File | Purpose |
|---|---|
| `checkpoint_utils.py` | `CheckpointTracker`: atomic checkpoint saves, best-model symlink management, resume-from-checkpoint |
| `train_utils.py` | Shared epoch runner, optimizer factory (AdamW with weight-decay split), `WafCollator` instantiation, per-epoch evaluation loop |

---

## Stage 6 — Distillation & Compression

**Directory:** `stages/6_distillation_and_compression/`  
**Make target:** `make distill_train`

Compresses the 99M teacher into a deployable 10M student and exports it for inference.

### Scripts

#### `00_train_teacher_99m.py`

Re-entry point that trains (or resumes) the 99M teacher if not already present. Delegates to Stage 5 logic.

#### `01_student_arch.py`

Architecture validation before training begins:

- Verifies vocab and dimension consistency between teacher and student
- Prints parameter breakdown: embeddings, attention, FFN, head
- Guards compression ratio: warns if ratio < 5× or > 50×

#### `02_distill_train.py`

Knowledge distillation training loop using `DistillationTrainer`:

- Hard-target CE loss weighted by `α_hard=0.3`
- Soft-target KL divergence on teacher logits at temperature T=4.0, weighted by `α_soft=0.7`
- Optional hidden-state MSE when `w_mse > 0`
- Temperature scheduling during training
- Early stopping on validation loss

#### `03_student_calibrate.py`

Post-training temperature calibration on the validation set. Optimises a scalar temperature parameter to minimise Expected Calibration Error (ECE). Saves calibrated temperature to the student checkpoint.

#### `04_student_canary.py`

Evaluates the student on the canary split (2% of data held completely out of training). Detects memorisation: a high canary detection rate with low FPR indicates genuine generalisation rather than overfitting.

#### `05_export_and_bench.py`

Exports the calibrated student and benchmarks the exported artefacts:

| Export format | Notes |
|---|---|
| ONNX | Dynamic axes; opset 17; validated with `onnxruntime` |
| TorchScript | `torch.jit.trace` for deployment without Python runtime |

Benchmarks reported: p50 / p95 / p99 latency at batch=1 (inline) and batch=64 (offline), throughput (RPS), peak VRAM, activation memory. Results written to `reports/export_bench.json`.

---

## Stage 7 — Evaluation

**Directory:** `stages/7_evaluation/`  
**Make targets:** `make eval_detection`, `make eval_latency`, `make eval_adversarial`, `make eval_ablation`, `make eval_interp`, `make compare_all`, `make report`  
**Full reference:** [docs/stage7_evaluation.md](stage7_evaluation.md)

Comprehensive evaluation suite covering detection efficacy, latency, adversarial robustness, ablations, interpretability, and artifact publishing.

### Scripts

#### Detection & Latency

| Script | Purpose |
|---|---|
| `01_detection_metrics.py` | Full metric suite (F1, AUC-PR, AUC-ROC, FPR, precision, recall) on the test split for all models; per-class breakdown; FP categorisation by Shannon entropy; K-Means cluster analysis on FP embeddings |
| `02_latency_bench.py` | p50/p95/p99/p99.9 latency + throughput at batch sizes [1, 8, 32, 64, 256] on GPU and CPU; PCIe transfer overhead measured separately; SLO PASS/FAIL verdict |
| `03_memory_footprint.py` | Peak VRAM, RSS RAM, and checkpoint sizes for all models |

#### Adversarial Robustness

| Script | Purpose |
|---|---|
| `04_evasion_payloads.py` | Evasion rate against all eight tamper transforms in `TAMPER_REGISTRY`; reports per-tamper and per-model |
| `05_obfuscation_robustness.py` | Robustness to chained multi-tamper sequences |
| `06_novel_attack_generalization.py` | Grammar-fuzzed novel attack families (500–1000 variations per class); reports detection rate with 95% Wilson score CIs |

#### Ablations

| Script | Purpose |
|---|---|
| `07_tokenizer_ablation.py` | Track A vs Track B OOV/fertility/performance; probe + truncated Transformer "Validation Anchor" corrects for proxy gap |
| `08_augmentation_ablation.py` | Metric delta attributable to synthetic augmentation; LR probe isolates the variable |
| `09_model_size_scaling.py` | AUC-PR vs. parameter count scaling curve; reads existing checkpoint metrics, no retraining |
| `10_label_smoothing_ablation.py` | Effect of label smoothing on calibration (ECE) and F1 |

#### Interpretability

| Script | Purpose |
|---|---|
| `11_attention_visualization.py` | Attention head heatmaps per attack class; head specialisation analysis; JSON output for offline rendering |
| `12_shap_analysis.py` | KernelSHAP token importance on the student model (CPU); requires `pip install shap` |
| `13_error_analysis.py` | FP/FN characterisation and confusion breakdown by attack class |

#### Reporting & Publishing

| Script | Purpose |
|---|---|
| `14_comparison_table.py` | Master table: all models × all metrics; `master_comparison_table.{csv,json}` |
| `15_generate_report.py` | Consolidates all eval JSON into `reports/final_evaluation_report.json` |
| `16_deployment_recommendation.py` | Weighted scoring (AUC-PR 50%, latency 30%, robustness 20%) → deployment decision; exits 1 if no model passes all SLOs |
| `17_push_to_hub.py` | Pushes all artifacts to HuggingFace Hub (`make push_hub` / `make push_hub_dry`) |

---

## Data Flow Between Stages

```
Stage 1  →  data/normalized/*.parquet
Stage 2  →  data/splits/{train,val,test,adversarial,canary}.parquet
             reports/slos.json
             reports/metrics/baselines.json
Stage 3  →  data/splits/ (augmented, re-split)
Stage 4  →  tokenizers/track_a/
             tokenizers/track_b/
Stage 5  →  models/teacher/{track_a_large,track_a_small}/best_model
Stage 6  →  models/track_b/99m/best_99m.pt
             models/student/best_student.pt
             models/student/student.onnx
Stage 7  →  reports/metrics/*.json
             reports/metrics/master_comparison_table.{csv,json}
             reports/final_evaluation_report.json
             reports/metrics/deployment_recommendation.json
```

All intermediate artefacts use the `HttpRecord` / Parquet schema defined in `ai_waf_v2/data/schema.py`. All training and evaluation metrics are logged to MLflow; run `make ui` to browse them at `localhost:5000`.
