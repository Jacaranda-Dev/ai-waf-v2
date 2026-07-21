# Architecture

> **📖 Docs:** [Index](README.md) · [User Guide](USER_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [API](API.md) · [Stages](stages/stages.md) · [Model Card](MODEL_CARD.md) · [Repo README](../README.md)

This document describes the big-picture design: how the pipeline stages connect,
how the core library is organised, and the key decisions behind both. For how to
*run* it, see [USER_GUIDE.md](USER_GUIDE.md); for the API surface, see
[API.md](API.md).

---

## 1. Two halves: a library and a pipeline

The repository is split into:

- **`ai_waf_v2/`** — a self-contained Python library (model, tokenizer, data,
  distillation, eval, utils). It has no GPU, network, or dataset dependency at
  import time, so it is unit-testable with synthetic data.
- **`stages/`** — seven numbered stage directories of standalone scripts that
  import the library and orchestrate the actual research workflow.

Everything the scripts share lives in the library; the scripts themselves only do
orchestration, I/O, and logging. This keeps the core logic testable and the
stages thin.

---

## 2. The pipeline DAG

Stages are strictly sequential; later stages consume earlier artefacts.

```
Stage 1  Acquire & curate ─► data/normalized/*.parquet
            │  download · normalize to HttpRecord · cross-dataset dedup · report
            ▼
Stage 2  Baselines        ─► data/splits/ · reports/2_baselines/metrics/01_slos.json · baseline metrics
            │  stratified split · SLO definitions · XGBoost · ModSecurity CRS
            ▼
Stage 3  Augmentation     ─► data/splits/ (augmented, re-split)
            │  grammar/mutator/tamper/LLM synthesis · quality gate · taxonomy
            ▼
Stage 4  Tokenization     ─► tokenizers/track_a/ · tokenizers/track_b/
            │  Track A: augment BERT vocab   Track B: custom BPE from scratch
            ▼
Stage 5  Teacher training ─► models/track_b/99m/best_99m.pt
            │  99M WafEncoder from scratch (+ DeBERTa Track A variants)
            ▼
Stage 6  Distillation     ─► models/student/ (calibrated, ONNX, TorchScript)
            │  KD → ~10M student · temperature calibration · canary · export+bench
            ▼
Stage 7  Evaluation       ─► reports/ (tables, figures, eval + deployment report)
               detection · latency · adversarial · ablations · interpretability
```

Each script is **idempotent** — it checks for its declared outputs via
`ai_waf_v2/utils/pipeline.py` (`require_inputs`, `check_output`) and skips unless
`--force` is given. This makes the pipeline resumable and safe to re-run.

Because stage directories begin with a digit (`5_teacher_training`), Python cannot
import them as modules. Shared classes that must be importable across scripts in a
stage are factored into non-numeric files (e.g.
`stages/5_teacher_training/track_b_model.py`).

---

## 3. The canonical data format

Every stage reads and writes the same schema, defined once in
`ai_waf_v2/data/schema.py`:

- **`HttpRecord`** (Pydantic) — one HTTP request with fields `id`, `method`,
  `path`, `query_string`, `headers`, `body`, `raw`, `label` (0 benign / 1
  malicious), `attack_class`, `source`, `split`.
- **`PARQUET_SCHEMA`** (PyArrow) — the on-disk representation. All intermediate
  data is Parquet, streamed in batches so multi-GB corpora never need to fit in
  RAM.

```
Raw HTTP ─► HttpRecord ─► Parquet ─► augmentation ─► quality filter
        ─► stratified split ─► WafDataset ─► training
```

`WafDataset` (map-style) and `WafDatasetMmap` (memory-mapped Arrow IPC) turn
Parquet into PyTorch datasets, tokenizing on the fly.

---

## 4. Core library layout

```
ai_waf_v2/
├── data/        schema (HttpRecord), datasets, dynamic-padding collator
├── augment/     recursive PCFG engine + label-neutral fillers + per-class validity checks
├── tokenizer/   custom HTTP-aware BPE (Track B) + pretrained augmentation (Track A)
├── models/      WafEncoder (from scratch), classifier head, distillation student
├── distill/     weighted CE + KL + hidden-MSE loss, distillation trainer
├── eval/        metrics, latency benchmarks, adversarial tamper suite
└── utils/       config loader, LLM abstraction, MLflow, logging, seed, timing, hub
```

See [API.md](API.md) for the public surface of each module.

---

## 5. Key design decisions

### HTTP-aware tokenization (two tracks)

Standard subword tokenizers fragment URLs, payloads, and encodings poorly.

- **Track B (primary)** — a byte-level BPE trained from scratch with a pre-tokenizer
  that treats `? & = / . ; : () {} []` as boundaries. Byte-level means *no UNK
  tokens* for arbitrary binary payloads. Vocab 8k, seq len 256.
- **Track A** — augments a pretrained BERT/DeBERTa vocabulary with HTTP-specific
  tokens, trading exact-fit for pretraining signal.

Stage 7 ablates the two on identical model weights to quantify the trade-off.

### Custom encoder, then distillation

The teacher (`WafEncoder`, ~99M: d_model 768, 13 layers, 12 heads) is trained from
scratch rather than fine-tuned, so the architecture is tuned to HTTP rather than
natural language. Attention uses `F.scaled_dot_product_attention`, auto-dispatching
to Flash Attention 2 on Ampere+ GPUs. Layers are pre-LayerNorm.

It is then distilled into a ~10M student (d_model 256, 6 layers) for deployment.
The loss combines hard-label cross-entropy (α=0.3), temperature-scaled KL against
the teacher (α=0.7, T=4.0), and an optional hidden-state MSE through a projection
layer. The T² factor keeps gradient magnitudes stable across temperatures.

### Operating point and SLOs

The model is optimized and reported at **FPR = 0.001** (false positives are
expensive for a WAF). Latency SLOs: p99 < 5 ms inline (batch 1), < 50 ms offline
(batch 64). These targets live in `config/pipeline.yaml` and are enforced by
Stage 7's `check_slo()` verdicts.

### Configuration

All hyperparameters live in `config/pipeline.yaml`, loaded through Pydantic models
in `ai_waf_v2/utils/config.py` with `${ENV_VAR}` expansion. Config validates split
ratios sum to 1.0 and `d_model % n_heads == 0`. There are no magic numbers in code,
so an experiment is fully described by its config plus its MLflow run.

### Synthetic data with realistic framing

Augmentation (Stage 3) doesn't just emit raw payloads — it fits method mix, UA
fingerprints, and header distributions from PCAP traces and applies them to both
attack re-framing and benign generation, preventing the model from learning to
detect "synthetic-looking" requests. Attack payloads come from recursive PCFG
grammars (`ai_waf_v2/augment/pcfg.py`, defined as data in
`config/pcfg_grammars.yaml`) that reach nested/variable-width structures, falling
back to flat templates for classes without a grammar. Incidental values in those
payloads (hosts, ports, ids, JS bodies) are `§NAME§` fillers drawn from
`ai_waf_v2/augment/fillers.py` — the *same* high-cardinality generators used for
benign traffic — so the model can't shortcut on a constant like `evil.com`; it's
the payload-level counterpart of the shared `HttpMetadataDistribution`. A five-pass
quality gate
(format validation, tokenizer UNK rate, per-class structural validity, MinHash-LSH
dedup, CRS label consistency) plus a leakage guard (Jaccard > 0.70 against
test/canary) filters the output.

---

## 6. Observability

Training and evaluation runs are logged to MLflow
(`ai_waf_v2/utils/mlflow_utils.py`); config is auto-flattened to dot-notation
params. Step timings can be captured with `utils/timing.py::StepTimer`. Logging is
centralised through `utils/logging.py` (Rich console + rotating file).
