# Stage 1 — Data Acquisition & Curation

> **📖 Docs:** [Index](../README.md) · [User Guide](../USER_GUIDE.md) · [Architecture](../ARCHITECTURE.md) · [API](../API.md) · [All Stages](stages.md) · [Model Card](../MODEL_CARD.md)

**Directory:** `stages/1_data_acquisition_and_curation/`  
**Make target:** `make data_collect`  
**Outputs:** `data/normalized/`, `reports/1_data_acquisition_and_curation/metrics/`

---

## Overview

Stage 1 downloads five public HTTP datasets, normalizes every record to the canonical `HttpRecord` Pydantic schema, deduplicates across sources, and produces a corpus health report that feeds the augmentation targeting logic in Stage 3.

The three scripts run sequentially and are idempotent by default — each checks whether its primary output file exists before doing any work.

| Script | Role |
|---|---|
| `01_acquire_and_normalize.py` | Download → normalize → write Parquet |
| `02_cross_dataset_dedup.py` | Exact SHA-256 + MinHash LSH deduplication |
| `03_generate_corpus_report.py` | Single-pass corpus health report (5 sections) |

---

## Scripts

### `01_acquire_and_normalize.py` — Download & normalize

**Inputs:** remote dataset URLs (config), `config/pipeline.yaml`  
**Outputs:** `data/normalized/{name}.parquet`, `data/normalized/all_datasets.parquet`, `reports/1_data_acquisition_and_curation/metrics/01_download_manifest.json`, `normalization_stats.json`, `collection_stats.json`

#### Dataset specs

Each dataset is described by a `DatasetSpec` dataclass carrying a name, download URL (with optional SHA-256 fingerprint), file type, and the name of its adapter function. The adapter registry (`ADAPTER_REGISTRY`) maps each dataset name to a generator function that converts raw rows to `HttpRecord` objects.

Three built-in adapters:
- `csic_v1` — CSIC 2010: CSV with absolute URLs (`http://localhost:8080/...`), attack class in a free-text `type` column
- `sr_bh_v1` — SR-BH 2020: two-column CSV (`request`, `label`); no per-class labels
- `http_params_v1` — ECML/PKDD 2007: parameterized query strings, class in a `classification` column

Additional datasets declared in `config/pipeline.yaml` under `data.datasets` are loaded via the converter registry (`CONVERTER_REGISTRY`) before the adapter is applied.

#### `_normalize_record()`

Applied to every record produced by any adapter:

1. **Null cleanup** — replaces `None` values with empty strings across all string fields
2. **Class canonicalization** — `_canonical_class()` maps raw label strings through `CLASS_ALIASES` to one of the six canonical class names (`sqli`, `xss`, `lfi`, `ssrf`, `cmdi`, `benign`); anything unrecognized becomes `unknown`
3. **Label enforcement** — `label=1` if `attack_class != "benign"`, `label=0` otherwise; prevents label/class mismatches from slipping through
4. **Raw rebuild** — `HttpRecord.build_raw()` reconstructs the full HTTP/1.1 request string from parsed fields, producing a canonical `raw` column regardless of the original format

#### Streaming write — `_write_batched()`

Records are accumulated in-memory in batches of 10 000 and flushed to Parquet with Snappy compression. This keeps peak RAM bounded regardless of dataset size. The PyArrow `PARQUET_SCHEMA` is enforced at write time; a schema mismatch raises immediately rather than silently coercing types.

#### Download pipeline — `acquire_csv()`

1. Download via `httpx` (streaming, progress bar)
2. Verify SHA-256 checksum if one is declared in the spec
3. Extract ZIP/GZIP archives if needed
4. Convert with the appropriate adapter
5. Write per-dataset Parquet to `data/normalized/{name}.parquet`
6. Append to the `all_datasets.parquet` accumulator

Kaggle-hosted datasets use `kagglehub` for download with an HTTP mirror fallback. When `KAGGLE_USERNAME` and `KAGGLE_KEY` are absent, the mirror fallback is tried automatically.

#### CLI flags

| Flag | Effect |
|---|---|
| `--force` | Re-download and re-normalize even if Parquet already exists |
| `--force-ingest` | Re-normalize from cached raw files without re-downloading |
| `--no-verify` | Skip SHA-256 verification (useful if checksum is stale) |
| `--only DATASET` | Process only the named dataset |

---

### `02_cross_dataset_dedup.py` — Cross-dataset deduplication

**Inputs:** `data/normalized/all_datasets.parquet`  
**Outputs:** `data/normalized/deduped.parquet`, `reports/1_data_acquisition_and_curation/metrics/02_dedup_stats.json`, `data/normalized/dedup_samples.txt`

Deduplication runs in two independent passes. It is a standalone stage (not merged into script 01) because MinHash LSH over millions of records is the most memory-intensive step in the curation pipeline; isolating it here lets the OS reclaim its working set before reporting starts.

#### Pass 1 — Exact SHA-256 deduplication

`_exact_dedup()` scans the table once, computing `SHA-256(raw)` for each record. Identical fingerprints (same raw HTTP string, modulo encoding) are dropped in O(n) time. Up to three example pairs are saved to `dedup_samples.txt` for manual inspection.

#### Pass 2 — MinHash LSH near-deduplication

`_minhash_dedup()` catches records that are textually similar but not identical — for example, CSIC 2010 and SR-BH 2020 payloads that differ only in whitespace or minor header variation.

Key implementation details:

- **Fingerprint field** — hashes `path + "?" + query_string + "\n" + body` rather than the full `raw` string. Headers (especially `User-Agent` and `Accept`) are excluded because shared header values inflate Jaccard similarity across unrelated requests from the same dataset.
- **CSIC 2010 URL normalization** — CSIC stores absolute URLs (`http://localhost:8080/path`). Before hashing, `urlparse().path` strips the scheme and host, which would otherwise make every CSIC pair share a 21-character prefix and produce artificially high Jaccard scores.
- **HTTP method filter** — near-duplicate candidates are restricted to those with the same HTTP method. A GET and POST to the same CSIC 2010 form endpoint share all parameters (query vs body) and would otherwise exceed the Jaccard threshold despite being distinct interactions.
- **Short-payload guard** — records with fewer than 40 unique character 3-grams skip MinHash entirely; short paths produce unreliable estimates and trigger false positives where a long shared prefix dominates.
- **128 permutations** — gives ~3% Jaccard estimation error at the configured threshold.

Default Jaccard threshold: `0.85` (from `cfg.data.dedup.minhash_threshold`). Override with `--threshold`.

#### CLI flags

| Flag | Effect |
|---|---|
| `--threshold FLOAT` | Override MinHash Jaccard threshold |
| `--exact-only` | Skip Pass 2 (useful for smoke tests) |
| `--force-dedup` | Re-run even if `deduped.parquet` already exists |

---

### `03_generate_corpus_report.py` — Corpus health report

**Inputs:** `data/normalized/deduped.parquet` (falls back to `all_datasets.parquet`)  
**Outputs:** five JSON files in `reports/1_data_acquisition_and_curation/metrics/`

This script consolidates five former scripts (`04_dataset_analysis.py`, `05_datasheet.py`, `06_taxonomy_coverage.py`, `07_length_distribution.py`, `08_taxonomy_inventory.py`) into a single invocation that reads the Parquet file exactly once.

The key design constraint: all five sections share the same in-memory DataFrame `df`. The only exception is `_report_datasheet()`, which reads only from config and never touches `df`. This means a single Parquet I/O operation pays for all five analyses.

#### Sections

| Section | Output file | Purpose |
|---|---|---|
| `dataset_analysis` | `dataset_analysis.json` | Record counts, imbalance ratio, length percentiles (min/mean/p95/p99/max), method distribution, per-source and per-class counts |
| `taxonomy_inventory` | `taxonomy_inventory.json` | Per-class counts, missing/low-count flags; **consumed by Stage 3.1** to compute augmentation gaps |
| `taxonomy_coverage` | `taxonomy_coverage.json` | Class × source coverage matrix (malicious records only); identifies which datasets contribute each attack class |
| `length_distribution` | `length_distribution.json` | Character and estimated token-length percentiles; `pct_requests_over_limit_approx` uses the BPE approximation chars/4 |
| `datasheet` | `datasheet.json` | Gebru et al. (2018) datasheet stub for the corpus |

#### BPE length approximation

`length_distribution` estimates token counts as `chars / 4`. This is an empirical approximation for HTTP payloads — the Track B tokenizer's actual fertility on validation data is the authoritative number (see Stage 4). The estimate is clearly labeled `bpe_approx` in the output.

#### `taxonomy_inventory.json` and augmentation

The `taxonomy_inventory.json` produced here is what `01_attack_synthesis.py` in Stage 3 reads to determine how many samples each attack class still needs. Classes listed in `cfg.data.data_schema.attack_classes` but absent from the corpus appear under `missing_classes`; classes below `cfg.augmentation.min_samples_per_class` appear under `low_count_classes`.

#### Selective re-runs

```bash
# Re-run only the two most expensive sections without touching the others
python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py \
    --section taxonomy_inventory --section length_distribution
```

`--section` can be repeated; the script validates the names against `ALL_SECTIONS` before loading the Parquet file.

---

## Data flow

```
[Remote datasets / Kaggle]
        │
        ▼
01_acquire_and_normalize.py
        │
        ├──▶ data/normalized/csic_2010.parquet
        ├──▶ data/normalized/sr_bh_2020.parquet
        ├──▶ ...
        └──▶ data/normalized/all_datasets.parquet
                    │
                    ▼
        02_cross_dataset_dedup.py
                    │
                    ├──▶ data/normalized/deduped.parquet          ← primary corpus
                    └──▶ data/normalized/dedup_samples.txt        ← example pairs
                                │
                                ▼
                03_generate_corpus_report.py
                                │
                                ├──▶ reports/1_data_acquisition_and_curation/metrics/03_dataset_analysis.json
                                ├──▶ reports/1_data_acquisition_and_curation/metrics/03_taxonomy_inventory.json  ← Stage 3 reads this
                                ├──▶ reports/1_data_acquisition_and_curation/metrics/03_taxonomy_coverage.json
                                ├──▶ reports/1_data_acquisition_and_curation/metrics/03_length_distribution.json
                                └──▶ reports/1_data_acquisition_and_curation/metrics/03_datasheet.json
```

---

## Configuration

Relevant keys in `config/pipeline.yaml`:

```yaml
data:
  datasets:
    - name: csic_2010
      url: "https://..."
      sha256: "abc123..."
      adapter: csic_v1
    # ...

  data_schema:
    attack_classes: [sqli, xss, lfi, ssrf, cmdi, benign]

  dedup:
    minhash_threshold: 0.85

augmentation:
  min_samples_per_class: 1000

paths:
  data_normalized: data/normalized
  data_splits:     data/splits
  reports:         reports
```

---

## Running the stage

```bash
# Full stage (download → dedup → report)
make data_collect

# Individual scripts
python stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py --config config/pipeline.yaml
python stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py --config config/pipeline.yaml
python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py --config config/pipeline.yaml

# Exact dedup only (skip MinHash — fast smoke test)
python stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py --exact-only

# Stricter MinHash threshold
python stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py --threshold 0.90

# Re-run only specific report sections
python stages/1_data_acquisition_and_curation/03_generate_corpus_report.py \
    --section taxonomy_inventory --section dataset_analysis
```

---

## What to check after running

| Check | Where |
|---|---|
| Per-dataset record counts | `reports/1_data_acquisition_and_curation/metrics/03_dataset_analysis.json` → `source_counts` |
| Dedup removal rate | `reports/1_data_acquisition_and_curation/metrics/02_dedup_stats.json` → `dedup_rate` |
| Example near-duplicate pairs | `data/normalized/dedup_samples.txt` |
| Missing attack classes | `reports/1_data_acquisition_and_curation/metrics/03_taxonomy_inventory.json` → `missing_classes` |
| Low-count classes | `reports/1_data_acquisition_and_curation/metrics/03_taxonomy_inventory.json` → `low_count_classes` |
| Truncation risk (pre-tokenization) | `reports/1_data_acquisition_and_curation/metrics/03_length_distribution.json` → `pct_requests_over_limit_approx` |
| Class × source coverage | `reports/1_data_acquisition_and_curation/metrics/03_taxonomy_coverage.json` → `coverage_matrix` |
| MLflow | `make ui` → experiments `01_acquire_and_normalize`, `02_cross_dataset_dedup`, `03_generate_corpus_report` |

A high `pct_requests_over_limit_approx` (>10%) at this stage is a forward-looking warning for the tokenizer window in Stage 4. The actual truncation rate — measured on tokenized sequences — is reported there.

---

◀ _(first stage)_ · [All Stages ▲](stages.md) · [Stage 2 — Baselines ▶](stage2_baselines.md)
