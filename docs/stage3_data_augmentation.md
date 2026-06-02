# Stage 3 — Data Augmentation

**Directory:** `stages/3_data_augmentation/`  
**Make target:** `make data_augment_all`  
**Outputs:** `data/augmented/`, `data/filtered/`, `data/splits/` (final), `reports/metrics/`

---

## Overview

Stage 3 synthesizes additional training samples to fill taxonomy gaps and balance class distributions. It runs in seven numbered steps that form a pipeline from raw payload generation through quality filtering and final split production.

| Script | Module | Role |
|---|---|---|
| `01_attack_synthesis.py` | A | Generate raw attack payloads + wrap in stub HTTP envelopes |
| `02_traffic_profiler.py` | B | Extract header/UA distributions from PCAP traces |
| `03_request_framing.py` | C | Re-frame attacks + generate benign traffic with realistic headers |
| `04_quality_gate.py` | D | Four-pass quality filter (format, UNK rate, dedup, label consistency) |
| `05_augmentation_probe.py` | E | Verify augmented samples improve generalization (CharCNN probe model) |
| `06_taxonomy_inventory.py` | F | Re-inventory attack class counts post-augmentation |
| `07_stratified_split.py` | G | Final stratified split on augmented + filtered corpus |

Scripts 02 and 03 share a design invariant: the same `HttpMetadataDistribution` singleton applies to BOTH attack re-framing and benign generation, preventing the model from learning to distinguish labels based on synthetic metadata fingerprints.

---

## Scripts

### `01_attack_synthesis.py` — Attack synthesis engine

**Inputs:** `data/splits/train.parquet` (seed payloads), `reports/metrics/taxonomy_inventory.json` (gap analysis)  
**Outputs:** `data/augmented/synthesis/synthesized_attacks.parquet`, `reports/metrics/augmentation_synthesis.json`

#### AugmentationGovernor

`AugmentationGovernor` is the orchestration layer. It loads `taxonomy_inventory.json`, computes `gaps()` (classes below `target_per_class`), and dispatches generation to a pool of generators in parallel using `ThreadPoolExecutor` (one thread per attack class).

`gaps()` returns `{attack_class: n_needed}` for every class in `GRAMMAR_REGISTRY` that is below the target. Classes at or above the target produce no output.

#### Generator types

All generators implement `BaseGenerator.generate_payloads(attack_class, n, rng) -> list[str]`, returning raw payload strings (not HTTP requests).

**`GrammarGenerator`** — Samples from `GRAMMAR_REGISTRY`, a registry of nine `AttackGrammar` dataclasses covering sqli, xss, lfi, ssrf, cmdi, path_traversal, header_injection, xxe, and ssti. Each grammar defines:
- `prefixes`, `payloads`, `suffixes` — assembled into a payload string by random selection
- `params`, `endpoints`, `methods` — used by the HTTP wrapper, not the payload itself

**`EncoderGenerator`** — Applies one of nine encoding mutations to a seed payload:

| Encoding | Technique |
|---|---|
| `url_encode` | Full percent-encoding |
| `double_url_encode` | Percent-encode the percent-signs |
| `partial_url_encode` | 50% chance to encode each non-alphanumeric character |
| `hex_encode` | Replace alpha chars with `0x{hex}` literals |
| `unicode_escape` | Replace alpha chars with `\u{codepoint}` |
| `html_entity` | Replace alpha chars with `&#{codepoint};` |
| `comment_insertion` | Replace whitespace with `/**/` |
| `case_variation` | Random per-character case flip |
| `whitespace_bypass` | Replace spaces with tab, LF, CR, double-space, or their encoded equivalents |

Seeds are drawn from `_load_seed_payloads()` — the query strings and bodies of existing malicious records in the training split.

**`ObfuscatorGenerator`** — Applies class-specific syntax transforms. Each attack class has its own dict of named transforms:

| Class | Example transforms |
|---|---|
| sqli | `apostrophe_mask`, `modsecurity_safe` (`OR` → `\|\|`), `space2dash`, `space2mssqlblank` |
| xss | `tag_case` (`<script` → `<Script`), `backtick_exec` (`alert(1)` → `` alert`1` ``), `null_byte_event` |
| lfi | `double_encode`, `overlong_utf8`, `dotdotslash` (`../` → `....//`), `null_byte` |
| cmdi | `ifs_space` (space → `${IFS}`), `brace_expand`, `quote_break`, `hex_cmd` |
| ssrf | `ip_decimal`, `ip_octal`, `ip_hex`, `proto_confusion`, `ipv6_mapped` |

`STACKABLE` defines a safe secondary obfuscation that can be applied on top of the primary one without producing conflicts:
- `lfi`: `null_byte` (appends `%00` after any traversal transform)
- `ssrf`: `proto_confusion` (rewrites scheme after any IP transform)

**`LlmGenerator`** — Generates payloads by calling `call_llm()`. Each attack class has a dedicated prompt in `PROMPTS`. Parsing: expects a JSON array; falls back to line-by-line extraction if JSON is malformed. `_replace_placeholders()` substitutes textbook domain names (`evil.com`, `attacker.com`, etc.) with realistic alternatives from `_REALISTIC_DOMAIN_POOL`. Consecutive failure tracking: aborts after 5 failures in a row.

#### Mutation chain — `_build_chain()`

Every seed payload (from GrammarGenerator or LlmGenerator) passes through a deterministic transform chain before being wrapped in an HTTP record:

```
seed payload
  → ObfuscatorGenerator   (class-specific syntax obfuscation)
  → (stackable pass, 30% probability)
  → EncoderGenerator      (encoding-level mutation)
  → (second encoding, 30% probability — produces double-encoded variants)
```

The chain ensures that grammar seeds are never recorded verbatim; every output has been through at least one obfuscation and one encoding transformation.

#### LLM budget split

When both `LlmGenerator` and `GrammarGenerator` are active, the gap is split by `llm_ratio` (default 0.50):
- 50% of needed samples requested from LLM (higher diversity, slower)
- 50% from grammar (deterministic, instant, no API cost)

If the LLM falls short of its quota, the grammar fills in with random duplicates of existing seeds.

#### HTTP envelope

`_payload_to_record()` wraps each raw payload in a minimal but schema-valid HTTP envelope using `_CLASS_META` (correct method/param/endpoint per class). Headers are stubs (`Host: stub.invalid`, `User-Agent: stub`) — these are replaced by script 03 during re-framing. The stub prevents script 01 from depending on the PCAP distribution that script 02 produces.

---

### `02_traffic_profiler.py` — Traffic distribution profiler

**Inputs:** PCAP files from `cfg.augmentation.benign.replay_path` (optional)  
**Outputs:** `reports/metrics/traffic_distribution.json`, `reports/metrics/traffic_profiler.json`

`PcapMetadataExtractor` parses HTTP flows from PCAP/PCAPNG files using `dpkt` (preferred) or `scapy` (fallback). It extracts per-flow dictionaries with method, URI, header dict, and body.

`TrafficDistribution.from_flows()` aggregates frequency distributions:

| Distribution | Purpose |
|---|---|
| `user_agents` | Top-50 UA strings by frequency |
| `methods` | GET/POST/PUT/DELETE/... weights |
| `path_depths` | Top-10 URL segment depths |
| `content_types` | Top-50 Content-Type values |
| `accept_headers` | Top-50 Accept values |
| `has_auth` | Fraction of flows with Authorization header |
| `has_referer` | Fraction of flows with Referer header |

When no PCAP is available (`replay_enabled: false` or path not found), `TrafficDistribution.from_flows([])` returns a zero distribution and script 03 uses its internal hardcoded defaults. The fallback is logged explicitly.

---

### `03_request_framing.py` — HTTP request framing engine

**Inputs:** `data/augmented/synthesis/synthesized_attacks.parquet`, `reports/metrics/traffic_distribution.json`  
**Outputs:** `data/augmented/framed/framed_records.parquet`, `reports/metrics/request_framing.json`

This script is the source of truth for HTTP request structure across the entire augmented corpus. Both attack re-framing and benign generation draw from the same endpoint registry, the same header distribution, and the same injection mechanics.

#### `HttpMetadataDistribution` singleton — `DIST`

Module-level singleton configured at startup from `traffic_distribution.json`. When PCAP data is available, it overrides internal defaults for UA, method, Content-Type, Accept, auth probability, and referer probability. When not available, internal defaults apply.

Header components are generated combinatorially rather than sampled from small fixed lists:

- **User-Agent** — 5 browser/tool families × multiple OS strings × multiple version strings → hundreds of unique UAs
- **Referer** — 36% chance of search-engine referer (with query term), 64% chance of in-app URL
- **Accept** — browser-style or API-style depending on whether the request has a body
- **Accept-Language** — random primary locale + optional secondary with q-value

#### `_ENDPOINT_REGISTRY`

A single registry of ~100 REST endpoint tuples `(method, path_template, natural_params)` used by both attack and benign framing. Both labels draw from the same pool, so the model cannot distinguish attack from benign based on URL or request structure alone.

#### Injection site dispatch — `_SITE_DISPATCH`

`frame_record()` selects an injection site according to per-class weighted probabilities defined in `_ATTACK_META`, then calls the corresponding framing function:

| Site | Description |
|---|---|
| `query_param` | Payload placed in a query string parameter |
| `post_body` | Payload placed in a form-encoded or JSON POST body |
| `path_segment` | Payload placed in a URL path component (`{id}` slot) |
| `cookie` | Payload placed in a cookie value alongside innocent cookies |
| `json_nested` | Payload nested inside a JSON body with innocent wrapper keys |
| `multipart` | Payload placed in a multipart filename field |
| `host_header` | Payload placed in the Host header |
| `header` | Payload placed in a class-specific header (e.g., Location for header_injection) |
| `raw_body` | Payload placed directly as the raw body (used for xxe) |

`_fill_params()` fills all non-injection parameters with type-appropriate innocent values from `_PARAM_VALUE_MAP` (a dict of 70+ named parameter generators covering pagination, search, user fields, dates, prices, etc.). This produces realistic surrounding context for every attack record.

#### Benign generation

**Programmatic (`_make_benign_record()`)** — Samples a random endpoint from `_ENDPOINT_REGISTRY`, fills all parameters with innocent values via `_fill_params()` (no injection slot), and assembles headers from `DIST`. Produces `n_programmatic` records in a loop.

**LLM-based (`generate_llm_benign()`)** — Generates realistic HTTP request dicts via `call_llm()` using six benign prompts covering mobile e-commerce, browser API calls, auth flows, GraphQL, and admin dashboards. LLM is used for benign framing only — never for generating raw attack payloads.

**Edge-case benign** — A separate LLM generation pass targeting hard negatives: search queries containing SQL keywords in natural English, HTML in POST bodies that is not XSS, and legitimate relative paths that look like traversal. These are tagged `attack_class=benign_edge_case` so the quality gate handles them specially.

The `benign_llm_ratio` (default 0.33) determines the fraction of the benign budget allocated to LLM generation; the rest is programmatic.

---

### `04_quality_gate.py` — Quality filtering pipeline

**Inputs:** `data/augmented/framed/framed_records.parquet`, `data/normalized/deduped.parquet`  
**Outputs:** `data/filtered/filtered.parquet`, `data/filtered/rejected.parquet`, `data/filtered/quarantined.parquet`, `reports/metrics/quality_gate.json`

The quality gate is a `QualityPipeline` that runs four passes in sequence. Passes 1 and 2 execute in parallel (stateless); Passes 3 and 4 execute sequentially (stateful/policy).

Only the final framing output is loaded — `data/augmented/synthesis/synthesized_attacks.parquet` is intentionally excluded because it contains the same attack payloads as `framed_records.parquet` but with stub headers, so loading both would double-count every attack record.

#### Pass 1 — HTTP format validation (`FormatValidator`)

Stateless, thread-safe. Rejects records where:
- `raw` is shorter than 10 characters
- First token is not a valid HTTP method (`GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS`)
- `raw` is longer than 8 192 characters

#### Pass 2 — Tokenizer UNK rate (`TokenizerCoverage`)

Loads the Track B tokenizer (`HttpTokenizer.load()`). Computes the fraction of non-padding tokens that are `[UNK]`. Rejects records where this rate exceeds `cfg.augmentation.filtering.max_unk_ratio`. Skipped gracefully if the Track B tokenizer has not been trained yet.

#### Pass 3 — Semantic deduplication (`SemanticDedup`)

Stateful MinHash LSH (threshold from `cfg.data.dedup.semantic_threshold`). Processes records sequentially; each kept record is inserted into the LSH index before the next record is evaluated. Records with no near-neighbors are kept; those with any near-neighbor are rejected as `semantic_duplicate`.

#### Pass 4 — CRS-aligned label consistency (`LabelConsistency`)

Compares each record's assigned label against a `CRSHeuristic` pattern match (same patterns as `02_modsecurity_crs.py` PL1 rules, without anomaly scoring). Three outcomes:

| Record | CRS match | Source | Action |
|---|---|---|---|
| `label=0` (benign) | Yes | not edge-case | **Quarantine** — likely mislabeled or CRS FP; excluded from training, written to `quarantined.parquet` for manual review |
| `label=0` (benign) | Yes | edge-case | **Flag + keep** — expected (deliberate hard negative); `source` gets `_crs_flagged` suffix |
| `label=1` (attack) | No | any | **Flag + keep** — evasive attack that CRS missed; `source` gets `_crs_flagged` suffix |
| anything | expected | any | **Pass** |

The two flagged-but-kept categories are counted separately in `pass4_stats`:
- `n_flagged_evasive` — attacks that evade CRS (high-value training samples)
- `n_flagged_edge` — edge-case benign records that trigger CRS rules (hard negatives)

#### Cross-augmentation leakage guard

After all four passes, any record with Jaccard similarity > 0.70 to `test.parquet` or `canary.parquet` is removed. The holdout LSH index is built from both files before the pipeline runs. This prevents the model from being evaluated on data it has effectively seen during training.

#### Quarantine review

Records in `quarantined.parquet` require manual review (see `notebooks/03_quarantine_review.ipynb`). Each quarantined record is a `label=0` record that triggered a CRS rule — it is either a genuine CRS false positive (valuable hard negative, should be restored to training) or a mislabeled malicious record (should be re-labeled or discarded).

---

### `05_augmentation_probe.py` — Augmentation quality probe

**Inputs:** `data/splits/train.parquet` (pre-aug), `data/filtered/filtered.parquet` (post-aug), `data/splits/val.parquet`  
**Outputs:** `reports/metrics/augmentation_probe.json`

Trains a lightweight `CharCNNProbe` model (1D-CNN, 16 384 character features, stride-4 conv, average pool) first without augmented data and then with it. The improvement in validation AUC-PR is the empirical evidence that augmentation helps generalization.

Fixed after a CUDA OOM bug: the original design allocated a `[256, 128, 65536]` activation tensor (~8.6 GiB). Fixes applied:
- `n_features` reduced from 65 536 to 16 384
- `conv1` stride changed from 1 to 4, reducing the feature map to `[batch, 128, 4096]` (~268 MB)
- `AdaptiveMaxPool1d` replaced with `AdaptiveAvgPool1d` (removes a non-determinism warning on CUDA)

---

### `06_taxonomy_inventory.py` — Post-augmentation inventory

**Inputs:** `data/splits/train.parquet` (augmented), `reports/metrics/taxonomy_inventory.json`  
**Outputs:** `reports/metrics/taxonomy_inventory_post_aug.json`

Re-runs the taxonomy inventory logic from Stage 1.3 on the augmented training split. Confirms that the augmentation gaps targeted by script 01 have been filled and that no class has dropped below the minimum threshold.

Fixed a double-nested `getattr` bug that always returned the fallback value of 5000 regardless of config:

```python
# Before (broken):
target = getattr(getattr(cfg.augmentation, None, None), "target_per_class", 5_000)
# After:
target = getattr(cfg.augmentation, "target_per_class", 5_000) if hasattr(cfg, "augmentation") else 5_000
```

---

### `07_stratified_split.py` — Final stratified split

**Inputs:** `data/filtered/filtered.parquet`  
**Outputs:** `data/splits/{train,val,test,adversarial,canary}.parquet` (overwritten), `reports/metrics/split_stats.json`

Re-runs the same stratified split logic as `00_stratified_split.py` (Stage 2) on the augmented and filtered corpus, replacing the pre-augmentation splits in `data/splits/`. The same sequential split algorithm, same fallback behavior, and same MLflow logging apply.

Fixed a bug where `pq.Table.from_pandas()` was called on `pq` (`pyarrow.parquet`) instead of `pa` (`pyarrow`). The fix adds the missing `import pyarrow as pa` and changes the call to `pa.Table.from_pandas()`.

---

## Data flow

```
taxonomy_inventory.json ──▶ 01_attack_synthesis.py
data/splits/train.parquet ─┘       │
                                   ▼
                       data/augmented/synthesis/
                       synthesized_attacks.parquet
                                   │
PCAP traces ──▶ 02_traffic_profiler.py ──▶ traffic_distribution.json
                                                │
                       ┌────────────────────────┘
                       │
                       ▼
             03_request_framing.py
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
  attack re-framed  benign REST  benign LLM
         └─────────────┴─────────────┘
                       │
                       ▼
          data/augmented/framed/
          framed_records.parquet
                       │
data/normalized/deduped.parquet ──▶ 04_quality_gate.py
                       │                   │
                       │     ┌─────────────┼──────────────┐
                       │     ▼             ▼              ▼
                       │  filtered.    rejected.    quarantined.
                       │  parquet      parquet      parquet
                       │                   │
                       │           ┌───────┘
                       │           ▼
                05_augmentation_probe.py ──▶ augmentation_probe.json
                06_taxonomy_inventory.py ──▶ taxonomy_inventory_post_aug.json
                07_stratified_split.py   ──▶ data/splits/ (final)
```

---

## Configuration

Relevant keys in `config/pipeline.yaml`:

```yaml
augmentation:
  target_per_class:      5000   # Minimum samples per attack class
  chain_length:          2
  double_encode_prob:    0.30
  double_obfuscate_prob: 0.30
  llm_ratio:             0.50   # Fraction of attack budget from LLM
  benign_llm_ratio:      0.33   # Fraction of benign budget from LLM
  min_samples_per_class: 1000

  llm:
    provider:        ollama     # anthropic | google | local | ollama
    model:           llama3.1
    model_path:      ""         # GGUF path for provider=local
    ollama_base_url: http://localhost:11434
    max_tokens:      256
    max_tokens_benign: 1024
    temperature:     0.9
    request_timeout: 30
    samples_benign:  2000
    samples_edge_case: 500
    batch_size_benign: 3

  benign:
    rest_samples:    10000
    replay_enabled:  false
    replay_path:     ""

  filtering:
    max_unk_ratio:   0.15    # Tokenizer UNK rate threshold for Pass 2

  rules:
    encodings:       [url_encode, double_url_encode, hex_encode, unicode_escape]

data:
  dedup:
    semantic_threshold: 0.90   # MinHash LSH threshold for Pass 3 (quality gate)

paths:
  data_augmented: data/augmented
  data_filtered:  data/filtered
```

---

## Running the stage

```bash
# Full augmentation pipeline
make data_augment_all

# Individual scripts
python stages/3_data_augmentation/01_attack_synthesis.py --config config/pipeline.yaml
python stages/3_data_augmentation/02_traffic_profiler.py --config config/pipeline.yaml
python stages/3_data_augmentation/03_request_framing.py  --config config/pipeline.yaml
python stages/3_data_augmentation/04_quality_gate.py     --config config/pipeline.yaml
python stages/3_data_augmentation/05_augmentation_probe.py --config config/pipeline.yaml
python stages/3_data_augmentation/06_taxonomy_inventory.py --config config/pipeline.yaml
python stages/3_data_augmentation/07_stratified_split.py   --config config/pipeline.yaml

# Use a local GGUF model for attack synthesis
python stages/3_data_augmentation/01_attack_synthesis.py \
    --config config/pipeline.yaml \
    --model-path /path/to/llama-3.1.gguf

# Use a cloud provider for benign framing
python stages/3_data_augmentation/03_request_framing.py \
    --config config/pipeline.yaml \
    --provider anthropic

# Run quality gate with more parallel workers
python stages/3_data_augmentation/04_quality_gate.py --workers 8

# Upload framed records to HuggingFace Hub
python stages/3_data_augmentation/03_request_framing.py --save --revision v1.0
```

---

## What to check after running

| Check | Where |
|---|---|
| Gaps filled per class | `reports/metrics/augmentation_synthesis.json` → `gaps_filled` |
| LLM provider and model used | `reports/metrics/augmentation_synthesis.json` → `llm_provider`, `llm_model` |
| Attack vs benign framing counts | `reports/metrics/request_framing.json` |
| Quality gate rejection rate | `reports/metrics/quality_gate.json` → `rejection_rate` |
| Rejection breakdown by pass | `reports/metrics/quality_gate.json` → `rejection_reasons` |
| Quarantine count (requires review) | `reports/metrics/quality_gate.json` → `n_quarantined` |
| Evasive attacks flagged (kept) | `reports/metrics/quality_gate.json` → `flagged_kept.evasive_attack` |
| Leakage removals | `reports/metrics/quality_gate.json` → `n_leaked` |
| Post-aug class balance | `reports/metrics/taxonomy_inventory_post_aug.json` → `class_counts` |
| Probe AUC-PR lift | `reports/metrics/augmentation_probe.json` |
| Final split sizes | `reports/metrics/split_stats.json` |
| Quarantine review | `notebooks/03_quarantine_review.ipynb` |
| MLflow | `make ui` → experiments `01_attack_synthesis`, `03_request_framing`, `04_quality_gate`, etc. |

A non-zero `n_quarantined` triggers a warning log and requires manual review before proceeding to Stage 4. Skipping review means potentially mislabeled records in the training set.

A high Pass 3 rejection rate (semantic dedup) indicates that the grammar generator is producing too many near-identical variants. Consider reducing `chain_length` or enabling a wider set of encoding types in `augmentation.rules.encodings`.
