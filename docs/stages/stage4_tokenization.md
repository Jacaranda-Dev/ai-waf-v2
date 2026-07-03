# Stage 4 — Tokenization

> **📖 Docs:** [Index](../README.md) · [User Guide](../USER_GUIDE.md) · [Architecture](../ARCHITECTURE.md) · [API](../API.md) · [All Stages](stages.md) · [Model Card](../MODEL_CARD.md)

**Directory:** `stages/4_tokenization/`  
**Make targets:** `make tokenize_b` (Track B, primary path); Track A scripts run individually  
**Outputs:** `tokenizers/track_a/`, `tokenizers/track_b/`, `reports/4_tokenization/metrics/*.json`

---

## Overview

Stage 4 produces the tokenizer that every downstream model (teacher and student) uses to convert raw HTTP requests into integer token sequences. Two competing tokenization strategies are evaluated side-by-side:

| | Track A | Track B |
|---|---|---|
| **Base** | BERT-base-uncased (WordPiece, 30 522 tokens) | Trained from scratch on HTTP corpus |
| **Approach** | Add HTTP-specific tokens to an existing vocabulary | Byte-level BPE with HTTP-aware pre-tokenization |
| **Vocab size** | ~30 522 + N added | 8 000 |
| **Pretraining signal** | Yes (inherits BERT representations) | No |
| **HTTP awareness** | Partial (new tokens may still be shadowed) | Full (regex splits on `?`, `&`, `=`, `/`, etc.) |

The comparison report produced by script 05 is the decision point: it determines which track feeds into Stage 5 training.

---

## Shared Evaluation Module — `tokenizer_eval.py`

All metric computation flows through this shared library. Scripts 02, 04, and 05 import from it; script 03 used to duplicate the logic and no longer does.

### Metrics

| Metric | Definition | What it signals |
|---|---|---|
| **OOV rate** | Fraction of subword tokens that are `[UNK]` | Vocabulary coverage; high OOV → the tokenizer is guessing on attack payloads |
| **Fertility** | Total subword tokens / total characters | Fragmentation; high fertility → long sequences, more truncation risk |
| **Truncation rate** | Fraction of requests whose token count exceeds `seq_len` | Content loss; tail of long requests (where XSS/LFI payloads often live) is silently dropped |
| **Subword-char ratio** | Characters per subword token (inverse of fertility) | Compression efficiency |
| **Avg seq len** | Mean tokens per request (pre-truncation) | Capacity headroom relative to `seq_len=256` |

All five are computed globally and per-attack-class. Per-class breakdown is important: a tokenizer can look healthy overall while performing poorly on a specific class (e.g. high OOV on `cmdi` payloads that use shell metacharacters outside the vocabulary).

### Sampling strategy

Evaluation samples are drawn with `stratified_sample()` — equal representation per attack class, not proportional. This is intentional: benign traffic vastly outnumbers some attack classes, so proportional sampling would leave minority classes (SSRF, CMDI) with too few samples to produce meaningful per-class metrics. The trade-off is that aggregate numbers reflect an artificial distribution; interpret them alongside the per-class breakdown.

### Implementation detail — no double tokenization

`compute_full_metrics()` tokenizes each text once and passes the resulting `token_seqs` to `_compute_per_class_from_seqs()`, which partitions the pre-computed sequences by class. Earlier versions called `_tokenize_batch()` a second time inside `compute_per_class_metrics()`; this has been removed.

---

## Scripts

### `01_augment_pretrained_vocab.py` — Track A setup

**Inputs:** `data/splits/train.parquet`, `config/pipeline.yaml` (`tokenizer.track_a`)  
**Outputs:** `tokenizers/track_a/` (full HuggingFace tokenizer), `reports/4_tokenization/metrics/01_tokenizer_track_a.json`

Calls `augment_pretrained_vocab()` from `ai_waf_v2.tokenizer.vocab_utils`, which:
1. Loads BERT-base-uncased via HuggingFace `AutoTokenizer`
2. Adds the tokens listed under `tokenizer.track_a.http_tokens` in `config/pipeline.yaml`
3. Saves the extended tokenizer to `tokenizers/track_a/`

After augmentation, a **token shadowing analysis** is run on every added token:

```
UNION SELECT  → ["UNION SELECT"]       n_fragments=1  shadowed=False  ✓
WAITFOR DELAY → ["WAIT", "##FOR", …]  n_fragments=4  shadowed=True   ✗
```

WordPiece assigns scores to all possible decompositions. Adding a whole-token entry does not guarantee the tokenizer will prefer it — if the subword decomposition achieves a higher score, the token is "shadowed" and effectively unused. The report surfaces all shadowed tokens so they can be addressed (e.g. by adjusting tokenizer config or removing them from the list).

The shadowing summary is written into `tokenizer_track_a.json` and surfaced again in the comparison report produced by script 05.

---

### `02_measure_oov_track_a.py` — Track A evaluation

**Inputs:** `tokenizers/track_a/`, `data/splits/val.parquet`  
**Outputs:** `reports/4_tokenization/metrics/02_tokenizer_oov_track_a.json`

Loads the augmented tokenizer and computes the full metric schema on a stratified 5 000-sample draw from the validation split. Results are written to the canonical Track A metrics file that script 05 reads.

---

### `03_train_custom_bpe.py` — Track B training

**Inputs:** `data/splits/train.parquet`  
**Outputs:** `tokenizers/track_b/tokenizer.json`, `tokenizers/track_b/train_corpus.txt`

Two steps:

**1. Corpus construction**

Reads the training split, draws up to 500 000 samples (reproducible random shuffle), and writes one HTTP request per line. Before writing, HTTP protocol delimiters are replaced with dedicated special tokens:

```
\r\n  →  [CRLF]
\n    →  [LF]
\r    →  [CR]
```

CRLF must be checked before `\n` and `\r` to avoid double substitution. These tokens preserve the structural boundary between HTTP headers and body — information that raw whitespace collapsing would destroy. BPE can then learn merge rules that span the header/body boundary, which correlates with content injection attacks.

**2. BPE training**

`HttpTokenizer.train()` runs byte-level BPE on the corpus with:
- Vocab size: 8 000 (from config)
- HTTP-aware pre-tokenization: regex splits on `?`, `&`, `=`, `/`, `.`, `;`, `:`, `()`, `{}`, `[]`
- Byte-level fallback: no `[UNK]` token — any byte sequence is representable

Evaluation is **not** performed here; it is deferred to script 04, which is the canonical Track B evaluation step. Script 03 only logs training-time params (`vocab_size`, `corpus_sample_size`) to MLflow.

---

### `04_measure_oov_track_b.py` — Track B evaluation

**Inputs:** `tokenizers/track_b/`, `data/splits/val.parquet`  
**Outputs:** `reports/4_tokenization/metrics/04_tokenizer_oov_track_b.json`

Mirrors script 02 exactly but for Track B. Loads `HttpTokenizer` (custom BPE), draws a stratified 5 000-sample val split, runs `compute_full_metrics()`. This file is the one script 05 reads — not the stats file script 03 used to write.

---

### `05_compare_tokenizers.py` — side-by-side comparison

**Inputs:** `tokenizers/track_a/`, `tokenizers/track_b/`, cached metrics from scripts 02 & 04  
**Outputs:** `reports/4_tokenization/metrics/05_tokenizer_comparison.json`

**Cache-first evaluation:** if both `tokenizer_oov_track_a.json` and `tokenizer_oov_track_b.json` exist, their pre-computed metrics are used directly. Pass `--recompute` to force a live re-evaluation on a fresh stratified sample (3 000 samples).

**Vocab Jaccard overlap:** computes the intersection-over-union of the two vocabulary sets. Near-zero Jaccard is expected (BERT WordPiece vs. HTTP-specific BPE) and confirms the tracks are genuinely different rather than overlapping.

**Shadowing attachment:** if `tokenizer_track_a.json` exists (from script 01), the shadowing summary is embedded in the comparison output so the Track A shadowing penalty is visible alongside the metric comparison.

**Console summary:**

```
┌─────────────────────────┬────────────────┬────────────────┬──────────┐
│ Metric                  │    Track A     │    Track B     │  Winner  │
├─────────────────────────┼────────────────┼────────────────┼──────────┤
│ OOV rate                │         0.0312 │         0.0000 │ track_b  │
│ Fertility               │         0.8140 │         0.6920 │ track_b  │
│ Truncation rate         │         0.1230 │         0.0870 │ track_b  │
│ Avg seq len             │       209.4100 │       177.6300 │ track_b  │
│ Subword/char ratio      │         1.2300 │         1.4500 │ track_b  │
└─────────────────────────┴────────────────┴────────────────┴──────────┘
  Vocab Jaccard overlap : 0.0041
  Recommendation        : track_b
```

**Recommendation heuristic:** counts winners per metric; the track that wins a strict majority of the five metrics is recommended. Returns `inconclusive` if neither wins a majority (e.g. 3–2 split after rounding differences). The recommendation is logged to MLflow as a param so it is queryable across experiment runs.

---

## Data flow

```
data/splits/train.parquet
        │
        ├──▶ 01_augment_pretrained_vocab.py ──▶ tokenizers/track_a/
        │                                        tokenizer_track_a.json (shadowing)
        │
        └──▶ 03_train_custom_bpe.py
                 │
                 └── train_corpus.txt (500k samples, CRLF-normalised)
                          │
                          ▼
                     tokenizers/track_b/

data/splits/val.parquet
        │
        ├──▶ 02_measure_oov_track_a.py ──▶ tokenizer_oov_track_a.json
        └──▶ 04_measure_oov_track_b.py ──▶ tokenizer_oov_track_b.json
                                                      │
                                  ┌───────────────────┘
                                  │
                                  ▼
                        05_compare_tokenizers.py ──▶ tokenizer_comparison.json
```

---

## Configuration

All tokenizer settings live under `tokenizer` in `config/pipeline.yaml`:

```yaml
tokenizer:
  seq_len: 256          # Model input length; requests longer than this are truncated

  track_a:
    base_model: bert-base-uncased
    output_dir: tokenizers/track_a
    http_tokens:          # Added to BERT vocab; shadowing analysis checks each one
      - "UNION SELECT"
      - "OR 1=1"
      - "../"
      # ...

  track_b:
    vocab_size: 8000
    output_dir: tokenizers/track_b
```

---

## Running the stage

```bash
# Full Track B pipeline (primary path, used by make train_b_99m)
make tokenize_b

# Individual scripts
python stages/4_tokenization/01_augment_pretrained_vocab.py --config config/pipeline.yaml
python stages/4_tokenization/02_measure_oov_track_a.py
python stages/4_tokenization/03_train_custom_bpe.py
python stages/4_tokenization/04_measure_oov_track_b.py
python stages/4_tokenization/05_compare_tokenizers.py

# Force live recomputation of comparison metrics (ignores cached 02/04 outputs)
python stages/4_tokenization/05_compare_tokenizers.py --recompute

# Re-run any script unconditionally
python stages/4_tokenization/03_train_custom_bpe.py --force
```

All scripts must be run from the repo root (or any directory) — `sys.path` is patched at startup to locate `tokenizer_eval.py` regardless of working directory.

---

## What to check after running

| Check | Where |
|---|---|
| Shadowed tokens | `reports/4_tokenization/metrics/01_tokenizer_track_a.json` → `shadowing.summary.shadowed_tokens` |
| Per-class OOV for each track | `tokenizer_oov_track_a.json` / `tokenizer_oov_track_b.json` → `per_class` |
| Truncation rate | Both OOV reports → `truncation_rate`; warn if > 0.10 |
| Comparison recommendation | `tokenizer_comparison.json` → `recommendation` |
| MLflow | `make ui` → experiments `02_measure_oov_track_a`, `04_measure_oov_track_b`, `05_compare_tokenizers` |

A truncation rate above 10 % at `seq_len=256` means a meaningful fraction of requests are being silently clipped. Options: increase `seq_len` (memory cost quadratic with attention), reduce fertility by adjusting the BPE vocab size, or accept the loss and document it.

---

[◀ Stage 3 — Data Augmentation](stage3_data_augmentation.md) · [All Stages ▲](stages.md) · [Stage 5 — Teacher Training ▶](stage5_teacher_training.md)
