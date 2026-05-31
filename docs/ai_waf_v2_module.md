# `ai_waf_v2` — Module Reference

`ai_waf_v2` is the core library for the Transformer-based WAF classifier. All pipeline stage scripts import from it. It is fully self-contained: no GPU, network access, or dataset files are needed to import it.

---

## Directory Layout

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
    ├── logging.py          # Rich console + rotating file logger
    ├── mlflow_utils.py     # MLflow experiment helpers
    ├── pipeline.py         # Stage guards: require_inputs / check_output
    └── seed.py             # seed_everything()
```

---

## `data` — HTTP Record Schema and Datasets

### `data/schema.py`

Defines the canonical HTTP record format that every stage reads and writes.

#### `HttpRecord` (Pydantic model)

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

Key methods:

- `build_raw()` — reconstructs the raw HTTP/1.1 request string from structured fields
- `headers_dict()` — parses the JSON `headers` field into a plain dict
- `from_dict(d)` — tolerant constructor that accepts partial or improperly typed dicts

#### PyArrow schema and I/O helpers

```python
PARQUET_SCHEMA       # PyArrow schema used for all Parquet files
FEATURE_COLUMNS      # ["method", "path", "query_string", "headers", "body", "raw"]
LABEL_COLUMN         # "label"
META_COLUMNS         # ["id", "attack_class", "source", "split"]

records_to_table(records: list[HttpRecord]) -> pa.Table
table_to_records(table: pa.Table) -> list[HttpRecord]
```

All data files in `data/normalized/` and the augmented splits use this schema.

---

### `data/dataset.py`

PyTorch `Dataset` wrappers that tokenize on the fly.

#### `WafDataset`

Map-style dataset backed by Parquet.

```python
ds = WafDataset(
    parquet_path="data/splits/train.parquet",
    tokenizer=tokenizer,
    max_length=256,
)
item = ds[0]
# {"input_ids": Tensor, "attention_mask": Tensor,
#  "labels": Tensor, "attack_class": str}
```

`ds.class_weights` — inverse-frequency class weights for imbalanced training.

#### `WafDatasetMmap`

Memory-mapped variant for datasets that exceed available RAM. Backed by Feather v2 (Arrow IPC). Shares the same interface as `WafDataset`.

```python
parquet_to_ipc(parquet_path, ipc_path)   # one-time conversion
ds = WafDatasetMmap(ipc_path, tokenizer)
```

#### Utility functions

```python
get_split_path(data_dir, split_name)   # canonical path resolver
load_split_stats(parquet_path)         # label counts without full load
```

---

### `data/collator.py`

#### `WafCollator`

Dataclass collator. Pads each batch to its own longest sequence (not the global `max_length`), saving GPU memory.

Optional parameters:
- `include_teacher_logits` — injects pre-computed soft labels for offline distillation
- `include_attack_class` — passes per-sample class strings through to the batch dict

---

## `tokenizer` — HTTP-Aware BPE

### `tokenizer/http_tokenizer.py`

#### `HttpTokenizer` (Track B)

Custom BPE tokenizer trained from scratch on HTTP corpora.

**Design decisions:**
- Byte-level pre-tokenisation handles arbitrary binary payloads without UNK tokens for individual bytes
- HTTP-aware split regex treats `?`, `&`, `=`, `/`, `.`, `;`, `:`, `(`, `)`, `{`, `}`, `[`, `]` as token boundaries, so URL structure is explicitly represented in the token stream
- Vocabulary size: 8 000 (configurable)
- Maximum sequence length: 256

**Special tokens:**

| Token | ID |
|---|---|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[CLS]` | 2 |
| `[SEP]` | 3 |
| `[MASK]` | 4 |

**Interface:**

```python
tok = HttpTokenizer()
tok.train(corpus_paths, vocab_size=8000, save_dir="tokenizers/track_b/")

tok = HttpTokenizer.load("tokenizers/track_b/")
ids  = tok.encode("GET /search?q=1 OR 1=1--")
batch = tok.encode_batch(["...", "..."])
text  = tok.decode(ids)

tok.compute_oov_rate(samples)     # fraction of [UNK] tokens
tok.compute_token_fertility(samples)  # avg tokens per whitespace word
```

---

### `tokenizer/vocab_utils.py`

Track A utilities — augment an existing pretrained tokenizer instead of training from scratch.

```python
n_added = augment_pretrained_vocab(
    base_model="bert-base-uncased",
    http_token_file="config/http_tokens.txt",
    save_dir="tokenizers/track_a/",
)

oov_rate, avg_len, fertility = measure_oov(tokenizer, samples)

report = compare_tokenizers(
    tok_a, tok_b, samples,
    per_class=True,   # OOV by attack class
)
```

---

## `models` — Neural Architecture

### `models/encoder.py`

Custom Transformer encoder; no pretrained weights.

#### Architecture (Track B 99M default)

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

#### `MultiHeadSelfAttention`

Uses `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+), which dispatches to Flash Attention 2 on Ampere+ GPUs automatically. Padding masks are converted to additive attention bias.

#### `EncoderLayer`

Pre-LayerNorm block:

```
x → LayerNorm → MultiHeadSelfAttention → + x
  → LayerNorm → FeedForward             → + x
```

#### `WafEncoder`

```python
enc = WafEncoder(config)
hidden = enc(input_ids, attention_mask)   # (B, seq, d_model)

enc.count_parameters()          # int
enc.parameter_breakdown()       # dict per sub-module
```

Weights initialised with N(0, 0.02) (BERT convention).

---

### `models/head.py`

#### `ClassificationHead`

```
[CLS] hidden → Dropout → Linear(d_model, num_classes)
```

#### `WafClassifier`

Full model: `WafEncoder` + `ClassificationHead`.

```python
model = WafClassifier.from_config(config.model.track_b)

logits, loss, hidden = model(
    input_ids, attention_mask, labels=labels, return_hidden=True
)

probs, preds = model.predict(input_ids, attention_mask, threshold=0.5)
model.save("checkpoints/teacher/best.pt")
model = WafClassifier.load("checkpoints/teacher/best.pt")
```

---

### `models/student.py`

#### `StudentClassifier`

Same architecture as `WafClassifier` but smaller by default (d_model=256, 6 layers, 4 heads ≈ 10M parameters). Adds:

- Optional hidden-state projection layer to align dimensions with teacher for MSE loss
- `prepare_for_int8_quantization()` — replaces `nn.Linear` layers with `bnb.nn.Linear8bitLt` for QAT via bitsandbytes

```python
student = StudentClassifier.from_config(config.model.student)

logits, loss, hidden = student(
    input_ids, attention_mask, labels=labels,
    teacher_hidden=teacher_hidden,   # enables MSE loss
)

student.prepare_for_int8_quantization()
student.save("checkpoints/student/best.pt")
```

---

## `distill` — Knowledge Distillation

### `distill/losses.py`

#### `DistillationLoss`

Combined hard-label CE + soft-label KL + optional hidden-state MSE:

```
L = α_hard × CE(student_logits, hard_labels)
  + α_soft × T² × KL( softmax(student/T) ‖ softmax(teacher/T) )
  + w_mse  × MSE(proj(student_hidden), teacher_hidden)
```

Defaults: `α_hard=0.3`, `α_soft=0.7`, `T=4.0`. Validates `α_hard + α_soft == 1.0`.

The T² factor keeps gradient magnitudes consistent across temperatures.

```python
loss_fn = DistillationLoss(alpha_hard=0.3, alpha_soft=0.7, temperature=4.0)
loss = loss_fn(
    student_logits, teacher_logits, hard_labels,
    student_hidden=h_s, teacher_hidden=h_t,
)
```

#### `SoftCrossEntropyLoss`

Cross-entropy against soft (probability) targets. Used when hard labels are unavailable.

---

### `distill/trainer.py`

#### `DistillationTrainer`

Full training loop: teacher (frozen, eval mode) → student.

Features:
- AdamW with weight-decay applied only to non-bias, non-LayerNorm parameters
- `_WarmupCosineScheduler`: linear warmup then cosine decay
- bf16 mixed precision (`torch.amp.autocast`)
- Gradient accumulation and clipping
- MLflow metric logging every N steps
- Best checkpoint saved on validation AUC-PR

```python
trainer = DistillationTrainer(
    teacher, student, loss_fn,
    train_loader, val_loader,
    config=config.training.distillation,
    mlflow_run=run,
)
trainer.train()
```

---

## `eval` — Evaluation Suite

### `eval/metrics.py`

Binary classification metrics implemented without scikit-learn.

```python
metrics = compute_metrics(probs, labels, threshold=0.5)
# {"f1", "precision", "recall", "fpr", "fnr", "auc_roc", "auc_pr"}

per_class = compute_per_class_metrics(probs, labels, attack_classes)
# dict[attack_class -> metrics_dict]

sweep = compute_threshold_sweep(probs, labels, n_thresholds=200)
# list[{"threshold", "f1", "precision", "recall", "fpr"}]

threshold = find_threshold_at_fpr(probs, labels, target_fpr=0.001)
```

Primary operating point: **FPR = 0.001** (SLO from `config/pipeline.yaml`).

---

### `eval/latency.py`

#### `LatencyBenchmark`

PyTorch inference benchmark with synthetic random batches.

```python
bench = LatencyBenchmark(model, seq_len=256, device="cuda")
results = bench.run(batch_sizes=[1, 8, 32], n_warmup=50, n_runs=500)
# results[batch_size] -> {"p50_ms", "p95_ms", "p99_ms", "throughput_rps"}

bench.check_slo(target_p99_ms=5.0, target_rps=10_000)
bench.print_table()
bench.save_json("reports/latency.json")
```

#### `OnnxLatencyBenchmark`

Same interface, runs ONNX Runtime sessions instead of PyTorch forward.

```python
bench = OnnxLatencyBenchmark("checkpoints/student.onnx", seq_len=256)
```

---

### `eval/adversarial.py`

#### Tamper registry

Eight obfuscation / evasion functions applied to malicious payloads:

| Function | Technique |
|---|---|
| `tamper_space2comment` | Replace spaces with SQL comments (`/**/`) |
| `tamper_randomcase` | Random character casing |
| `tamper_url_encode` | Single URL encoding |
| `tamper_double_url_encode` | Double URL encoding |
| `tamper_hex_encode` | Hex literal encoding |
| `tamper_between_comments` | Insert SQL comments between characters |
| `tamper_base64_param` | Wrap parameters in base64 |
| `tamper_unicode_escape` | Unicode escape sequences |

#### `AdversarialEvaluator`

```python
evaluator = AdversarialEvaluator(model, tokenizer, device="cuda")
results = evaluator.evaluate(malicious_samples)
# list[AdversarialResult] — one per tamper function
# fields: tamper_name, detection_rate, evasion_rate, n_samples, failure_examples
```

---

## `utils` — Infrastructure

### `utils/config.py`

Typed configuration hierarchy (Pydantic v2). Loaded from `config/pipeline.yaml` with `${ENV_VAR}` expansion.

```python
config = load_config("config/pipeline.yaml")

config.project.name           # str
config.paths.data             # Path
config.slo.latency_inline_p99_ms   # float = 5.0
config.data.splits.train      # float = 0.70
config.tokenizer.track_b.vocab_size  # int = 8000
config.model.track_b.d_model  # int = 768
config.model.student.d_model  # int = 256
config.training.distillation.temperature  # float = 4.0
config.mlflow.experiment_name # str
```

Notable validations:
- `SplitConfig`: split ratios must sum to 1.0
- `ModelArchConfig`: `d_model % n_heads == 0`

---

### `utils/logging.py`

```python
from ai_waf_v2.utils import get_logger, configure_root

configure_root()           # silence transformers / datasets / PIL noise
log = get_logger(__name__) # Rich console + rotating file (10 MB × 3)
log.info("stage started")
```

Call `configure_root()` once at the entry point of each stage script.

---

### `utils/mlflow_utils.py`

```python
from ai_waf_v2.utils.mlflow_utils import init_experiment, mlflow_run, log_metrics_dict

init_experiment(config)

with mlflow_run(config, run_name="distill-v3") as run:
    log_metrics_dict({"loss": 0.12, "auc_pr": 0.987}, step=100)
    log_artifact_path("checkpoints/student/best.pt")
```

Config is flattened to dot-notation params (e.g. `training.distillation.lr = 3e-4`). Param values are truncated to 250 characters (MLflow limit).

---

### `utils/seed.py`

```python
from ai_waf_v2.utils import seed_everything

seed_everything(42, deterministic=False)
# Seeds: Python random, NumPy, torch CPU, torch CUDA, PYTHONHASHSEED
# deterministic=True enables cuDNN determinism (~5–10% throughput cost)
```

---

### `utils/pipeline.py`

Shared idempotency guards used by every stage entry-point script. Keeps guard logic in one place so individual scripts stay thin.

#### `require_inputs(inputs: dict[str | Path, str]) -> None`

Verifies that all prerequisite files exist before starting work. Exits with code 1 on the first missing file, printing the make target the user should run to produce it.

```python
from ai_waf_v2.utils.pipeline import require_inputs

require_inputs({
    "data/normalized/deduped.parquet": "make data_collect",
    "reports/metrics/taxonomy_inventory.json": "make data_analyze",
})
```

If `data/normalized/deduped.parquet` is absent, output is:

```
ERROR  Required input missing: data/normalized/deduped.parquet  →  run: make data_collect
```

#### `check_output(output: Path, force: bool, label: str = "") -> bool`

Returns `True` (skip) when the primary output file already exists and is non-empty and `force` is `False`. Returns `False` when the caller should proceed (file missing, file empty, or `force=True`).

```python
from ai_waf_v2.utils.pipeline import check_output
from pathlib import Path

if check_output(Path("reports/corpus_report.json"), args.force, "Corpus report"):
    return   # already done
# ... do work ...
```

If the output exists, output is:

```
INFO   Corpus report already exists (42,317 bytes) — skipping. Pass --force to re-run.
```

---

## Public API Summary

```python
# Data
from ai_waf_v2.data import (
    HttpRecord, PARQUET_SCHEMA, records_to_table, table_to_records,
    WafDataset, WafDatasetMmap, get_split_path, load_split_stats,
    WafCollator,
)

# Tokenizer
from ai_waf_v2.tokenizer import (
    HttpTokenizer, SPECIAL_TOKENS,
    augment_pretrained_vocab, measure_oov, compare_tokenizers,
)

# Models
from ai_waf_v2.models import (
    WafEncoder, EncoderLayer, MultiHeadSelfAttention,
    WafClassifier, ClassificationHead,
    StudentClassifier,
)

# Distillation
from ai_waf_v2.distill import (
    DistillationLoss, SoftCrossEntropyLoss,
    DistillationTrainer,
)

# Evaluation
from ai_waf_v2.eval import (
    compute_metrics, compute_per_class_metrics,
    compute_threshold_sweep, find_threshold_at_fpr,
    LatencyBenchmark, OnnxLatencyBenchmark,
    AdversarialEvaluator, TAMPER_REGISTRY,
)

# Utils
from ai_waf_v2.utils import load_config, seed_everything, get_logger, configure_root
```
