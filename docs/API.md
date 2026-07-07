# API Reference — `ai_waf_v2`

> **📖 Docs:** [Index](README.md) · [User Guide](USER_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [API](API.md) · [Stages](stages/stages.md) · [Model Card](MODEL_CARD.md) · [Repo README](../README.md)

The core library imported by all pipeline stages. It is self-contained: importing
it needs no GPU, network, or dataset files. This page summarises the public API;
for design rationale see [ARCHITECTURE.md](ARCHITECTURE.md).

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
from ai_waf_v2.utils import (
    load_config, call_llm, seed_everything, get_logger, configure_root,
)
```

---

## `data` — schema and datasets

### `HttpRecord` (Pydantic)

Canonical HTTP record. Fields: `id`, `method`, `path`, `query_string`, `headers`
(JSON string), `body`, `raw`, `label` (0/1), `attack_class`, `source`, `split`.
Key methods: `build_raw()`, `headers_dict()`, `HttpRecord.from_dict(d)` (tolerant
constructor).

### I/O helpers

```python
PARQUET_SCHEMA        # PyArrow schema for all Parquet files
FEATURE_COLUMNS       # ["method","path","query_string","headers","body","raw"]
records_to_table(records) -> pa.Table
table_to_records(table)   -> list[HttpRecord]
```

### Datasets

- `WafDataset` — map-style, Parquet-backed, tokenizes on the fly; `.class_weights`
  for imbalanced training.
- `WafDatasetMmap` — memory-mapped (Arrow IPC) variant for corpora exceeding RAM.
- `parquet_to_ipc(parquet_path, ipc_path)`, `get_split_path(...)`,
  `load_split_stats(...)`.
- `WafCollator` — pads each batch to its own longest sequence; optional
  `include_teacher_logits` and `include_attack_class`.

---

## `augment` — synthetic attack generation

Dependency-light helpers used by Stage 3 (imported by the digit-prefixed stage
scripts, unit-tested via `tests/test_all.py`).

### Recursive PCFG (`augment.pcfg`)

- `Pcfg(name, rules, start="START", max_depth=6, max_len=256)` — samples a
  probabilistic recursive context-free grammar; `.sample(rng) -> str`. Terminates
  by construction (past `max_depth`, only minimal-cost productions are used; the
  constructor rejects non-terminating grammars).
- `load_pcfg_grammars(path) -> dict[str, Pcfg]` — loads grammars from YAML
  (`config/pcfg_grammars.yaml`); returns `{}` if the file is absent.
- `PcfgSampler(registry, fallback, oversample=8).generate(attack_class, n, rng)`
  — samples the PCFG where present, delegates to `fallback` for classes without a
  grammar, and tops up from `fallback` when uniques run short.
- `T/N/F/OPENQ/CLOSEQ`, `PCFG_FUNCS` — grammar symbol constructors and the named
  callable-terminal registry referenced from YAML as `f:<name>`.

```python
from ai_waf_v2.augment import load_pcfg_grammars, PcfgSampler
registry = load_pcfg_grammars("config/pcfg_grammars.yaml")
sampler  = PcfgSampler(registry, fallback=flat_generator.generate_payloads)
payloads = sampler.generate("sqli", 100, rng)
```

### Fillers (`augment.fillers`)

Label-neutral, high-cardinality substitution for incidental values so the model
learns attack *structure*, not constants like `evil.com`/`4444`.

- `fill_placeholders(text, rng, strict=False) -> str` — replace `§NAME§` /
  `§NAME#TAG§` (coreference) placeholders from `FILLERS`. Unknown names are left
  untouched unless `strict=True` (which raises — used by tests).
- `FILLERS` — name → generator: `HOST IP PORT PATH PARAM IDENT STR INT COL WORD JS_BODY`.
- `placeholder_names(text) -> set[str]` — filler names referenced in a string.
- `scrub_textbook_hosts(text, rng) -> str` — replace textbook hosts (`evil.com`…)
  in LLM output with `HOST` draws.

The same generators fill benign traffic in Stage 3.3, so a filled token carries no
label signal — the payload-level analogue of `HttpMetadataDistribution`.

### Leakage measurement (`eval.leakage`)

Checks whether the dataset/model keys on incidental constants instead of structure.

- `token_leakage(texts, labels, min_df=5, top_k=50) -> LeakageReport` — per-token
  `P(malicious|token)`, lift, PMI and mutual information; each token flagged
  `incidental` (host/number/id-shaped) or structural. Frequent + near-deterministic
  incidental tokens are shortcut candidates.
- `counterfactual_auc_delta(predict_fn, texts, labels, rng) -> {auc, auc_swapped, delta}`
  — `swap_fillers` re-randomises incidental values; a small `delta` means the
  model relied on structure, a large one means it memorised constants.
- `swap_fillers`, `auc_roc`, `pretokenize` — building blocks (pure stdlib).

Backs Stage 3.8 (`make data_leakage`), which writes `reports/3_data_augmentation/metrics/08_token_leakage.json`
and a model-free counterfactual (swap fillers, recompute — filler-driven MI drops).

### Payload validity (`augment.validity`)

- `is_valid_for_class(attack_class, text) -> bool` — lightweight structural check
  (regex + bracket balance; no parser deps) that a payload matches its class.
  Backs the Stage 3.4 quality gate. Unknown classes always pass.
- `VALIDATORS` — the per-class validator registry.

---

## `tokenizer` — HTTP-aware BPE

### `HttpTokenizer` (Track B)

Byte-level BPE trained from scratch. Vocab 8000, max seq 256. Specials:
`[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`.

```python
tok = HttpTokenizer()
tok.train(corpus_paths, vocab_size=8000, save_dir="tokenizers/track_b/")
tok = HttpTokenizer.load("tokenizers/track_b/")
ids   = tok.encode("GET /search?q=1 OR 1=1--")
batch = tok.encode_batch(["...", "..."])
text  = tok.decode(ids)
tok.compute_oov_rate(samples)
tok.compute_token_fertility(samples)
```

### Track A helpers

```python
n_added = augment_pretrained_vocab(base_model="bert-base-uncased",
                                   http_token_file="config/http_tokens.txt",
                                   save_dir="tokenizers/track_a/")
oov_rate, avg_len, fertility = measure_oov(tokenizer, samples)
report = compare_tokenizers(tok_a, tok_b, samples, per_class=True)
```

---

## `models` — architecture

- **`WafEncoder`** — Transformer encoder from scratch (no pretrained weights).
  Defaults: vocab 8000, d_model 768, 13 layers, 12 heads, 4× FFN, seq 256, learned
  positional embeddings, `[CLS]` pooling, ~99M params. Uses
  `F.scaled_dot_product_attention` (Flash Attention 2 on Ampere+), pre-LayerNorm.
- **`WafClassifier`** — `WafEncoder` + `ClassificationHead`.
- **`StudentClassifier`** — smaller encoder (d_model 256, 6 layers ≈ 10M); optional
  hidden projection for MSE distillation; `prepare_for_int8_quantization()`.

```python
model = WafClassifier.from_config(config.model.track_b)
logits, loss, hidden = model(input_ids, attention_mask, labels=labels, return_hidden=True)
probs, preds = model.predict(input_ids, attention_mask, threshold=0.5)
model.save("checkpoints/teacher/best.pt")
```

---

## `distill` — knowledge distillation

### `DistillationLoss`

```
L = α_hard · CE(student, hard_labels)
  + α_soft · T² · KL(softmax(student/T) ‖ softmax(teacher/T))
  + w_mse  · MSE(proj(student_hidden), teacher_hidden)
```

Defaults `α_hard=0.3`, `α_soft=0.7`, `T=4.0`; validates `α_hard + α_soft == 1.0`.
`SoftCrossEntropyLoss` is exported for soft-target-only training.

### `DistillationTrainer`

Full loop (teacher frozen → student): AdamW with no-decay for bias/LayerNorm,
linear warmup + cosine decay, bf16, gradient accumulation + clipping, MLflow
logging, best checkpoint on val AUC-PR.

---

## `eval` — evaluation suite

```python
metrics   = compute_metrics(probs, labels, threshold=0.5)
            # {"f1","precision","recall","fpr","fnr","auc_roc","auc_pr"}
per_class = compute_per_class_metrics(probs, labels, attack_classes)
sweep     = compute_threshold_sweep(probs, labels, n_thresholds=200)
threshold = find_threshold_at_fpr(probs, labels, target_fpr=0.001)   # primary point
```

- `LatencyBenchmark` / `OnnxLatencyBenchmark` — p50/p95/p99 + throughput;
  `check_slo()`, `print_table()`, `save_json()`.
- `AdversarialEvaluator.evaluate(samples)` — runs the 8 functions in
  `TAMPER_REGISTRY` (comment obfuscation, random casing, URL/double-URL/hex
  encoding, base64 wrapping, unicode escapes), returning per-tamper
  `detection_rate` / `evasion_rate`.

---

## `utils` — infrastructure

| Module | Surface |
|---|---|
| `config.py` | `load_config()` reads `config/pipeline.yaml` with `${ENV}` expansion → Pydantic models |
| `llm.py` | `call_llm(provider=anthropic\|google\|local\|ollama, ...) -> str \| None`; set `LLM_DEBUG=1` to log response previews |
| `mlflow_utils.py` | `init_experiment`, `mlflow_run`, `log_metrics_dict`, `log_artifact_path` |
| `logging.py` | `configure_root()` (once per entry point), `get_logger(__name__)` |
| `seed.py` | `seed_everything(42, deterministic=False)` |
| `timing.py` | `StepTimer` — `.step(name)` context manager, `.timings()`, `.save()`, `.log_mlflow()` |
| `pipeline.py` | stage-script harness: `require_inputs(...)`, `check_output(out, force, label)` |
| `hub.py` | HuggingFace publishing: `push_folder(...)`, `push_text(...)` (needs `HF_TOKEN`) |

### `call_llm` providers

| Provider | Backend | Auth |
|---|---|---|
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `google` | Google GenerativeAI | `GOOGLE_API_KEY` |
| `local` | llama-cpp-python (GGUF, offline) | none (`model_path=...`) |
| `ollama` | Ollama HTTP `/api/chat` | none (`ollama_base_url=...`) |
