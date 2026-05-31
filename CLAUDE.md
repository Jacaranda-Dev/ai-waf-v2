# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ai-waf-v2** is a Transformer-based Web Application Firewall (WAF) classifier with knowledge distillation. It is a multi-stage ML research pipeline for training and evaluating neural networks to detect malicious HTTP requests.

## Commands

### Setup
```bash
make setup              # Minimal venv + core deps
make setup MODE=full    # Adds ONNX, TensorRT, bitsandbytes
make setup MODE=research  # Adds Jupyter kernel
```

### Quality Checks
```bash
make lint               # Ruff check + format
make typecheck          # Mypy on stages/
make quality            # Both lint and typecheck
```

### Testing
```bash
make test               # Full suite (tests/test_all.py)
make test_core          # Model + tokenizer tests only
make test_data          # Dataset + augmentation tests
make test_distill       # Distillation + metrics tests
pytest tests/test_all.py::TestClassName::test_method -v  # Single test
```

### Pipeline Stages (run in order)
```bash
make data_collect       # Stage 1: Download & normalize datasets
make baselines          # Stage 2: XGBoost/LightGBM/ModSecurity baselines
make data_augment_all   # Stage 3: All augmentation substages + split
make tokenize_b         # Stage 4B: Train custom BPE tokenizer
make train_b_99m        # Stage 5: Train 99M custom teacher model
make distill_train      # Stage 6: Distill to 10M student + quantization
make eval_detection     # Stage 7: Full evaluation suite
make compare_all        # Stage 7.14: Master comparison table
```

### Shortcuts
```bash
make all                # Full end-to-end pipeline
make track_b_full       # Custom tokenizer + training only
make eval_only          # Evaluate existing checkpoints
make ui                 # MLflow UI at localhost:5000
```

## Architecture

### Pipeline Stages
Seven sequential stages under `stages/`, each with numbered substage scripts:
1. **Data acquisition & curation** — downloads 5 public datasets, normalizes to Parquet
2. **Baselines** — TF-IDF + XGBoost/LightGBM + ModSecurity CRS
3. **Data augmentation** — 11 substages: encoding mutations, grammar payloads (SQLi/XSS/LFI/SSRF/CMDi), local/cloud LLM generation, benign traffic
4. **Tokenization** — Track A (augment pretrained BERT vocab) vs Track B (custom BPE from scratch)
5. **Teacher training** — Three configs: DeBERTa-base (337M), BERT-tiny, custom 99M
6. **Distillation & compression** — 10M student via KD, QAT vs PTQ, ONNX + TensorRT export
7. **Evaluation** — Detection efficacy, latency, adversarial robustness, ablations, interpretability

### Core Library (`ai_waf_v2/`)
Self-contained library imported by all stage scripts:

- **`data/schema.py`** — `HttpRecord` (Pydantic) + PyArrow `PARQUET_SCHEMA`; all data flows through this canonical format with fields: `id`, `method`, `path`, `query_string`, `headers`, `body`, `raw`, `label`, `attack_class`, `source`, `split`
- **`data/dataset.py`** — `WafDataset` and `WafDatasetMmap` (Parquet → PyTorch Dataset)
- **`tokenizer/http_tokenizer.py`** — `HttpTokenizer`: HTTP-aware pre-tokenization (splits on `?`, `&`, `=`, `/`, `.`, `;`, `:`, `()`, `{}`, `[]`), byte-level BPE, 8k vocab, seq_len=256
- **`models/encoder.py`** — `WafEncoder`: custom Transformer from scratch with learned positional embeddings and Flash Attention 2 support
- **`models/student.py`** — `StudentClassifier`: smaller encoder with optional hidden projection for MSE distillation loss
- **`distill/losses.py`** — `DistillationLoss`: weighted CE (α=0.3) + T-scaled KL (α=0.7, T=4.0) + optional hidden-state MSE
- **`eval/metrics.py`** — `compute_metrics()` and per-class metrics (F1, precision, recall, AUC-ROC/PR); primary operating point is FPR=0.001
- **`utils/config.py`** — Pydantic config models; `load_config()` reads `config/pipeline.yaml` with env var expansion

### Configuration
All hyperparameters live in `config/pipeline.yaml`. No magic numbers in code. Key values:
- Data splits: 70% train / 15% val / 10% test / 3% adversarial / 2% canary
- Track B encoder: d_model=768, 13 layers, 12 heads, vocab=8000
- Student: d_model=256, 6 layers, 4 heads, ~10M params
- SLOs: p99 latency <5ms inline / <50ms offline; FPR <0.001

### Dual Tokenization Tracks
- **Track A**: Augments a pretrained tokenizer (BERT/DeBERTa) with HTTP-specific tokens
- **Track B**: Trains BPE from scratch using HTTP-aware pre-tokenization regex; artifacts saved to `tokenizers/track_b/`

### Data Flow
Raw HTTP → `HttpRecord` → Parquet (in `data/normalized/`) → augmentation → quality filtering → stratified split → `WafDataset` → training

### MLflow
All training runs logged via `ai_waf_v2/utils/mlflow_utils.py`. Use `make ui` to browse experiments. Experiments are stored locally unless `MLFLOW_TRACKING_URI` is set.

### Testing Philosophy
All tests in `tests/test_all.py` use synthetic in-memory data — no GPU, network, or dataset downloads required. Tests cover: `HttpRecord` I/O, tokenizer encode/decode, dataset loading, metrics, model forward passes, and distillation loss.

---

## `ai_waf_v2` — Module Reference

`ai_waf_v2` is the core library for the Transformer-based WAF classifier. All pipeline stage scripts import from it. It is fully self-contained: no GPU, network access, or dataset files are needed to import it.

### Directory Layout

```
ai_waf_v2/
├── __init__.py             # Package version (0.1.0)
├── data/
│   ├── schema.py           # HttpRecord + PyArrow schema
│   ├── dataset.py          # WafDataset / WafDatasetMmap
│   └── collator.py         # WafCollator (dynamic padding)
├── tokenizer/
│   ├── http_tokenizer.py   # Custom BPE (Track B)
│   └── vocab_utils.py      # Pretrained vocab augmentation (Track A)
├── models/
│   ├── encoder.py          # WafEncoder (Transformer from scratch)
│   ├── head.py             # WafClassifier (encoder + head)
│   └── student.py          # StudentClassifier (distillation target)
├── distill/
│   ├── losses.py           # DistillationLoss / SoftCrossEntropyLoss
│   └── trainer.py          # DistillationTrainer
├── eval/
│   ├── metrics.py          # F1, AUC-ROC/PR, threshold sweep
│   ├── latency.py          # PyTorch + ONNX latency benchmarks
│   └── adversarial.py      # Tamper functions + AdversarialEvaluator
└── utils/
    ├── config.py           # Pydantic config + YAML loader
    ├── llm.py              # Unified LLM call abstraction (anthropic | google | local | ollama)
    ├── logging.py          # Rich console + rotating file logger
    ├── mlflow_utils.py     # MLflow experiment helpers
    └── seed.py             # seed_everything()
```

### `data` — HTTP Record Schema and Datasets

#### `data/schema.py`

Defines the canonical HTTP record format that every stage reads and writes.

**`HttpRecord` (Pydantic model)**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique record identifier |
| `method` | `str` | HTTP method (`GET`, `POST`, …) |
| `path` | `str` | URL path component |
| `query_string` | `str` | Raw query string (without `?`) |
| `headers` | `str` | JSON-serialised header dict |
| `body` | `str` | Request body |
| `raw` | `str` | Full reconstructed HTTP/1.1 request |
| `label` | `int` | `0` = benign, `1` = malicious |
| `attack_class` | `str` | `sqli`, `xss`, `lfi`, `ssrf`, `cmdi`, `benign`, … |
| `source` | `str` | Originating dataset name |
| `split` | `str` | `train`, `val`, `test`, `adversarial`, `canary` |

Key methods: `build_raw()`, `headers_dict()`, `from_dict(d)` (tolerant constructor).

**PyArrow schema and I/O helpers**

```python
PARQUET_SCHEMA       # PyArrow schema used for all Parquet files
FEATURE_COLUMNS      # ["method", "path", "query_string", "headers", "body", "raw"]
LABEL_COLUMN         # "label"
META_COLUMNS         # ["id", "attack_class", "source", "split"]

records_to_table(records: list[HttpRecord]) -> pa.Table
table_to_records(table: pa.Table) -> list[HttpRecord]
```

#### `data/dataset.py`

- **`WafDataset`** — map-style Parquet-backed dataset; tokenizes on the fly; `ds.class_weights` for imbalanced training
- **`WafDatasetMmap`** — memory-mapped variant (Feather v2 / Arrow IPC) for datasets exceeding RAM
- `parquet_to_ipc(parquet_path, ipc_path)` — one-time Parquet → IPC conversion
- `get_split_path(data_dir, split_name)`, `load_split_stats(parquet_path)` — path and stats helpers

#### `data/collator.py`

**`WafCollator`** — pads each batch to its own longest sequence (not global `max_length`). Optional `include_teacher_logits` for offline distillation and `include_attack_class` for per-class metrics.

---

### `tokenizer` — HTTP-Aware BPE

#### `tokenizer/http_tokenizer.py` — `HttpTokenizer` (Track B)

Custom BPE tokenizer trained from scratch. Key design choices:
- Byte-level pre-tokenisation — no UNK tokens for arbitrary binary payloads
- HTTP-aware split regex — `?`, `&`, `=`, `/`, `.`, `;`, `:`, `()`, `{}`, `[]` are token boundaries
- Vocab size: 8 000; max seq length: 256

Special tokens: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`, `[MASK]=4`

```python
tok = HttpTokenizer()
tok.train(corpus_paths, vocab_size=8000, save_dir="tokenizers/track_b/")
tok = HttpTokenizer.load("tokenizers/track_b/")
ids  = tok.encode("GET /search?q=1 OR 1=1--")
batch = tok.encode_batch(["...", "..."])
text  = tok.decode(ids)
tok.compute_oov_rate(samples)        # fraction of [UNK] tokens
tok.compute_token_fertility(samples) # avg tokens per whitespace word
```

#### `tokenizer/vocab_utils.py` (Track A)

```python
n_added = augment_pretrained_vocab(
    base_model="bert-base-uncased",
    http_token_file="config/http_tokens.txt",
    save_dir="tokenizers/track_a/",
)
oov_rate, avg_len, fertility = measure_oov(tokenizer, samples)
report = compare_tokenizers(tok_a, tok_b, samples, per_class=True)
```

---

### `models` — Neural Architecture

#### `models/encoder.py` — `WafEncoder`

Custom Transformer encoder; no pretrained weights.

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 8 000 |
| Hidden size (`d_model`) | 768 |
| Layers | 13 |
| Attention heads | 12 |
| FFN expansion | 4× |
| Max sequence length | 256 |
| Positional embeddings | Learned (BERT-style) |
| Pooling | `[CLS]` token |
| ~Parameters | 99 M |

`MultiHeadSelfAttention` uses `F.scaled_dot_product_attention` (PyTorch 2.0+) — auto-dispatches to Flash Attention 2 on Ampere+ GPUs. `EncoderLayer` uses pre-LayerNorm (`x → LN → Attn → +x → LN → FFN → +x`). Weights initialised with N(0, 0.02).

#### `models/head.py` — `WafClassifier`

`[CLS] hidden → Dropout → Linear(d_model, num_classes)`. Full model = `WafEncoder` + `ClassificationHead`.

```python
model = WafClassifier.from_config(config.model.track_b)
logits, loss, hidden = model(input_ids, attention_mask, labels=labels, return_hidden=True)
probs, preds = model.predict(input_ids, attention_mask, threshold=0.5)
model.save("checkpoints/teacher/best.pt")
```

#### `models/student.py` — `StudentClassifier`

Same architecture as teacher, smaller by default (d_model=256, 6 layers, 4 heads ≈ 10M params). Adds:
- Optional hidden-state projection layer for MSE alignment with teacher
- `prepare_for_int8_quantization()` — replaces `nn.Linear` with `bnb.nn.Linear8bitLt` for QAT

```python
student = StudentClassifier.from_config(config.model.student)
logits, loss, hidden = student(
    input_ids, attention_mask, labels=labels, teacher_hidden=teacher_hidden
)
student.prepare_for_int8_quantization()
```

---

### `distill` — Knowledge Distillation

#### `distill/losses.py` — `DistillationLoss`

```
L = α_hard × CE(student_logits, hard_labels)
  + α_soft × T² × KL( softmax(student/T) ‖ softmax(teacher/T) )
  + w_mse  × MSE(proj(student_hidden), teacher_hidden)
```

Defaults: `α_hard=0.3`, `α_soft=0.7`, `T=4.0`. Validates `α_hard + α_soft == 1.0`. The T² factor keeps gradient magnitudes consistent across temperatures. Also exports `SoftCrossEntropyLoss` for soft-target-only scenarios.

#### `distill/trainer.py` — `DistillationTrainer`

Full training loop (teacher frozen → student). Features: AdamW with no-decay for bias/LayerNorm, linear warmup + cosine decay LR, bf16 mixed precision, gradient accumulation and clipping, MLflow logging, best checkpoint on val AUC-PR.

```python
trainer = DistillationTrainer(
    teacher, student, loss_fn, train_loader, val_loader,
    config=config.training.distillation, mlflow_run=run,
)
trainer.train()
```

---

### `eval` — Evaluation Suite

#### `eval/metrics.py`

Binary classification metrics (no scikit-learn dependency).

```python
metrics   = compute_metrics(probs, labels, threshold=0.5)
           # {"f1", "precision", "recall", "fpr", "fnr", "auc_roc", "auc_pr"}
per_class = compute_per_class_metrics(probs, labels, attack_classes)
sweep     = compute_threshold_sweep(probs, labels, n_thresholds=200)
threshold = find_threshold_at_fpr(probs, labels, target_fpr=0.001)
```

Primary operating point: **FPR = 0.001** (SLO from `config/pipeline.yaml`).

#### `eval/latency.py`

- **`LatencyBenchmark`** — PyTorch inference; reports p50/p95/p99 and throughput; `check_slo()`, `print_table()`, `save_json()`
- **`OnnxLatencyBenchmark`** — same interface, runs ONNX Runtime sessions

#### `eval/adversarial.py`

Eight tamper functions in `TAMPER_REGISTRY`:

| Function | Technique |
|---|---|
| `tamper_space2comment` | SQL comment obfuscation (`/**/`) |
| `tamper_randomcase` | Random character casing |
| `tamper_url_encode` | Single URL encoding |
| `tamper_double_url_encode` | Double URL encoding |
| `tamper_hex_encode` | Hex literal encoding |
| `tamper_between_comments` | SQL comment insertion |
| `tamper_base64_param` | Base64 wrapping |
| `tamper_unicode_escape` | Unicode escape sequences |

`AdversarialEvaluator.evaluate(samples)` returns a list of `AdversarialResult` (one per tamper function) with `detection_rate`, `evasion_rate`, and `failure_examples`.

---

### `utils` — Infrastructure

#### `utils/config.py`

Pydantic v2 config hierarchy loaded from `config/pipeline.yaml` with `${ENV_VAR}` expansion. Key paths: `config.model.track_b.d_model`, `config.model.student.d_model`, `config.training.distillation.temperature`, `config.slo.latency_inline_p99_ms`. Validates split ratios sum to 1.0 and `d_model % n_heads == 0`.

#### `utils/llm.py`

Unified LLM call abstraction used by any pipeline stage that needs text generation. Returns the raw text response (`str | None`); JSON parsing and any further structure extraction are the caller's responsibility.

```python
from ai_waf_v2.utils.llm import call_llm

text = call_llm(
    provider    = "anthropic",   # anthropic | google | local | ollama
    system      = "You are ...",
    user        = "Generate ...",
    model       = "claude-sonnet-4-6",
    max_tokens  = 512,
    temperature = 0.7,
    # provider=local  → model_path="/path/to/model.gguf"
    # provider=ollama → ollama_base_url="http://localhost:11434"
)
```

| Provider | Backend | Auth |
|---|---|---|
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `google` | Google GenerativeAI | `GOOGLE_API_KEY` |
| `local` | llama-cpp-python (GGUF, offline) | none |
| `ollama` | Ollama HTTP `/api/chat` | none |

The `local` provider lazy-loads the GGUF weights once per `model_path` (module-level cache); subsequent calls reuse the loaded model. Configured via `augmentation.llm` in `config/pipeline.yaml` — relevant fields: `provider`, `model`, `temperature`, `model_path`, `ollama_base_url`.

**Debugging** — set `LLM_DEBUG=1` to log a truncated preview of every model response:

```bash
LLM_DEBUG=1 python stages/3_data_augmentation/02_request_framing.py --config config/pipeline.yaml
```

Each call emits one `DEBUG` line with the first 200 characters of the raw response (newlines escaped), e.g.:

```
DEBUG  [ollama] response preview: '["1 OR 1=1--", "admin\'--", "1; DROP TABLE users--", ...]'
```

#### `utils/logging.py`

```python
configure_root()           # silence transformers / datasets / PIL noise — call once per entry point
log = get_logger(__name__) # Rich console + rotating file (10 MB × 3 backups)
```

#### `utils/mlflow_utils.py`

```python
init_experiment(config)
with mlflow_run(config, run_name="distill-v3") as run:
    log_metrics_dict({"loss": 0.12, "auc_pr": 0.987}, step=100)
    log_artifact_path("checkpoints/student/best.pt")
```

Config is auto-flattened to dot-notation params. Values truncated to 250 chars (MLflow limit).

#### `utils/seed.py`

```python
seed_everything(42, deterministic=False)
# Seeds Python, NumPy, torch CPU/CUDA, PYTHONHASHSEED
# deterministic=True enables cuDNN determinism (~5–10% throughput cost)
```

---

### Public API Summary

```python
from ai_waf_v2.data import (
    HttpRecord, PARQUET_SCHEMA, records_to_table, table_to_records,
    WafDataset, WafDatasetMmap, get_split_path, load_split_stats, WafCollator,
)
from ai_waf_v2.tokenizer import (
    HttpTokenizer, SPECIAL_TOKENS,
    augment_pretrained_vocab, measure_oov, compare_tokenizers,
)
from ai_waf_v2.models import (
    WafEncoder, EncoderLayer, MultiHeadSelfAttention,
    WafClassifier, ClassificationHead, StudentClassifier,
)
from ai_waf_v2.distill import (
    DistillationLoss, SoftCrossEntropyLoss, DistillationTrainer,
)
from ai_waf_v2.eval import (
    compute_metrics, compute_per_class_metrics,
    compute_threshold_sweep, find_threshold_at_fpr,
    LatencyBenchmark, OnnxLatencyBenchmark,
    AdversarialEvaluator, TAMPER_REGISTRY,
)
from ai_waf_v2.utils import load_config, call_llm, seed_everything, get_logger, configure_root
```

---

## Pipeline Stages Reference

The pipeline is composed of seven sequential stages under `stages/`. Each stage is a numbered directory containing standalone Python scripts. Stages must be run in order; later stages depend on artefacts produced by earlier ones.

### Stage 1 — Data Acquisition & Curation

**Directory:** `stages/1_data_acquisition_and_curation/`  
**Make target:** `make data_collect`

Downloads public HTTP datasets, Normalizes them to the canonical `HttpRecord` format, deduplicates across sources, and produces a corpus report.

| Script | Purpose |
|---|---|
| `01_acquire_and_normalize.py` | Single-pass download → Normalize → Parquet writer. SHA-256 verification, `CLASS_ALIASES` schema canonicalisation, streams to `data/normalized/` in 10 000-record batches. Datasets: CSIC 2010, SR-BH 2020, ECML/PKDD 2007, plus any declared in `config/pipeline.yaml` |
| `02_cross_dataset_dedup.py` | Two-pass dedup: Pass 1 exact SHA-256 hash of `raw`; Pass 2 MinHash LSH at Jaccard 0.85. Logs removal counts to MLflow |
| `03_generate_corpus_report.py` | Single-pass corpus analysis: dataset stats, taxonomy inventory, class × source coverage matrix, length percentiles, Gebru et al. datasheet stub → `reports/corpus_report.{json,html}` |

---

### Stage 2 — Baselines

**Directory:** `stages/2_baselines/`  
**Make target:** `make baselines`

Establishes non-neural reference points against which the Transformer model is measured.

| Script | Purpose |
|---|---|
| `00_stratified_split.py` | Stratified train/val/test/adversarial/canary split by `(label × attack_class)`; fallback to random split for rare classes |
| `01_define_slos.py` | Writes SLO targets to `reports/slos.json`: p99 inline <5 ms, p99 offline <50 ms, FPR <0.001 |
| `02_modsecurity_crs.py` | OWASP CRS 3.3 with Paranoia Levels PL1–PL4; 19 rules, severity-weighted anomaly scoring; SLO PASS/FAIL verdicts |
| `03_classical_ml_baseline.py` | Three baselines: (A) Aho-Corasick string match, (B) HashedNGram + XGBoost (2²⁰ buckets, GPU, primary baseline), (C) no-HTTP-features ablation to quantify field-awareness lift. Optional LightGBM comparison |

---

### Stage 3 — Data Augmentation

**Directory:** `stages/3_data_augmentation/`  
**Make target:** `make data_augment_all`

Synthesises additional training samples to fill taxonomy gaps and balance class distributions.

| Script | Purpose |
|---|---|
| `01_attack_synthesis.py` | `AugmentationGovernor` reads `taxonomy_inventory.json`, computes per-class gaps, dispatches to generator registry: `GrammarGenerator` (context-free templates), `MutatorGenerator` (8 encoding transforms), `TamperGenerator` (7 SQLMap-style tampers), `LlmGenerator` (any provider via `call_llm`). Parallel via ThreadPoolExecutor |
| `02_traffic_profiler.py` | Extracts method mix, UA fingerprints, Accept/Content-Type headers, and auth/referer rates from PCAP traces (dpkt + scapy fallback); writes `traffic_distribution.json`. Falls back to empty profile when no PCAP is available; Stage 3.3 then uses its own internal defaults |
| `03_request_framing.py` | Wraps payloads in realistic HTTP envelopes. `HttpMetadataDistribution` singleton reads `traffic_distribution.json` and applies PCAP-fitted distributions (method weights, UA, Accept, Content-Type, auth/referer rates) to BOTH attack re-framing and benign generation, preventing synthetic fingerprinting. Cloud LLM used for benign-only generation via Anthropic/Google APIs |
| `04_quality_gate.py` | Four-pass filter: (1) HTTP format validation, (2) tokenizer UNK-rate check, (3) MinHash LSH dedup, (4) CRS label consistency. Leakage guard removes records with Jaccard >0.70 to test/canary splits |
| `05_augmentation_probe.py` | Lightweight probe model to verify synthetic samples improve generalisation |
| `06_taxonomy_inventory.py` | Recomputes taxonomy inventory post-augmentation; verifies gaps are closed |
| `07_stratified_split.py` | Re-runs stratified split on augmented corpus to produce final `data/splits/` files |

---

### Stage 4 — Tokenization

**Directory:** `stages/4_tokenization/`  
**Make targets:** `make tokenize_b` (Track B); Track A scripts run individually

| Script | Purpose |
|---|---|
| `01_augment_pretrained_vocab.py` | **Track A** — add HTTP tokens to BERT-base-uncased; shadow analysis; saves to `tokenizers/track_a/` |
| `02_measure_oov_track_a.py` | OOV rate and token fertility for Track A |
| `03_train_custom_bpe.py` | **Track B** — train byte-level BPE from scratch on HTTP corpus; saves to `tokenizers/track_b/` |
| `04_measure_oov_track_b.py` | OOV rate and token fertility for Track B |
| `05_compare_tokenizers.py` | Side-by-side: OOV rate, fertility, vocabulary Jaccard, per-class OOV breakdown |
| `tokenizer_eval.py` | Extended evaluation utilities; token shadowing analysis |

**Track A** benefits from pretraining signal. **Track B** vocabulary is tuned exactly to HTTP structure but trained from scratch.

---

### Stage 5 — Teacher Training

**Directory:** `stages/5_teacher_training/`  
**Make target:** `make train_b_99m`

| Script | Purpose |
|---|---|
| `01_track_a_large.py` | Fine-tune DeBERTa-v3-base (337M) with Track A tokenizer; MLflow experiment `track_a_large` |
| `02_track_a_small.py` | Fine-tune smaller DeBERTa variant for a lighter distillation target |
| `03_track_b_99m.py` | Train 99M `WafEncoder` from scratch with Track B tokenizer; curriculum learning (short sequences first) |
| `03b_distill_track_b.py` | Cross-track distillation: transfer Track A teacher knowledge into Track B student |
| `checkpoint_utils.py` | `CheckpointTracker`: atomic saves, best-model symlink, resume-from-checkpoint |
| `train_utils.py` | Shared epoch runner, AdamW factory (weight-decay split), `WafCollator`, evaluation loop |

---

### Stage 6 — Distillation & Compression

**Directory:** `stages/6_distillation_and_compression/`  
**Make target:** `make distill_train`

Compresses the 99M teacher into a deployable 10M student and exports it for inference.

| Script | Purpose |
|---|---|
| `00_train_teacher_99m.py` | Re-entry point: trains or resumes the 99M teacher if not already present |
| `01_student_arch.py` | Pre-flight checks: vocab/dimension consistency, parameter breakdown, compression ratio guard (warns if <5× or >50×) |
| `02_distill_train.py` | KD training: CE (α=0.3) + T-scaled KL (α=0.7, T=4.0) + optional hidden MSE; early stopping on val loss |
| `03_student_calibrate.py` | Post-training temperature calibration on val set; minimises ECE; saves scalar temperature to checkpoint |
| `04_student_canary.py` | Evaluates student on canary split to detect memorisation |
| `05_export_and_bench.py` | Exports to ONNX (opset 17, dynamic axes) and TorchScript; benchmarks p50/p95/p99 latency, throughput, VRAM → `reports/export_bench.json` |

---

### Stage 7 — Evaluation

**Directory:** `stages/7_evaluation/`  
**Make targets:** `make eval_detection`, `make eval_only`, `make compare_all`

#### Detection & Latency

| Script | Purpose |
|---|---|
| `01_detection_metrics.py` | F1, AUC-PR, AUC-ROC, FPR, precision, recall on test split; per-class breakdown; threshold at FPR=0.001 |
| `02_latency_bench.py` | p50/p95/p99 and throughput at batch=1 (inline) and batch=64 (offline); SLO verdict |
| `03_memory_footprint.py` | Peak VRAM and activation memory during inference |

#### Adversarial Robustness

| Script | Purpose |
|---|---|
| `04_evasion_payloads.py` | Detection rate against encoding-mutated payloads |
| `05_obfuscation_robustness.py` | Robustness to comment insertion, whitespace bypass, random casing |
| `06_novel_attack_generalization.py` | Generalisation to held-out attack classes; zero-shot and few-shot settings |

#### Ablations

| Script | Purpose |
|---|---|
| `07_tokenizer_ablation.py` | Track A vs Track B on same model weights |
| `08_augmentation_ablation.py` | With vs. without synthetic augmentation data |
| `09_model_size_scaling.py` | AUC-PR vs. parameter count: 8M / 16M / 32M / 50M / 99M |
| `10_label_smoothing_ablation.py` | Effect of label smoothing on calibration and F1 |

#### Interpretability & Reporting

| Script | Purpose |
|---|---|
| `11_attention_visualization.py` | Attention head heatmaps; head specialisation analysis |
| `12_shap_analysis.py` | SHAP feature importance for classical baseline comparison |
| `13_error_analysis.py` | Confusion breakdown by attack class; common misclassification patterns |
| `14_comparison_table.py` | Master table: all models × all metrics; Markdown and LaTeX output |
| `15_generate_report.py` | Aggregates all eval JSON into `reports/eval_report.{json,html}` |
| `16_deployment_recommendation.py` | Automated deployment checklist: model selection, SLO verdict, rollout strategy |

---

### Data Flow Between Stages

```
Stage 1  →  data/normalized/*.parquet
Stage 2  →  data/splits/{train,val,test,adversarial,canary}.parquet
             reports/slos.json  ·  reports/baselines/
Stage 3  →  data/splits/ (augmented, re-split)
Stage 4  →  tokenizers/track_a/  ·  tokenizers/track_b/
Stage 5  →  checkpoints/teacher/{track_a_large,track_a_small,track_b_99m}/best_model
Stage 6  →  checkpoints/student/best.pt (calibrated)
             checkpoints/student/model.onnx  ·  checkpoints/student/model.pt
Stage 7  →  reports/eval_report.{json,html}
             reports/comparison_table.{md,tex}
             reports/deployment_recommendation.md
```

All intermediate artefacts use the `HttpRecord` / Parquet schema from `ai_waf_v2/data/schema.py`. All training metrics are logged to MLflow; run `make ui` to browse at `localhost:5000`.
