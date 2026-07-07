# Attack Synthesis — Stage 3.1

> **📖 Docs:** [Index](../README.md) · [User Guide](../USER_GUIDE.md) · [All Stages](../stages/stages.md) · [Attack Synthesis](attack_synthesis.md)

**Script:** `stages/3_data_augmentation/01_attack_synthesis.py`  
**Output:** `data/augmented/synthesis/synthesized_attacks.parquet`  
**Make target:** part of `make data_augment_all`

---

## Overview

Attack synthesis fills taxonomy gaps by generating synthetic malicious HTTP payloads for each attack class that is under-represented in the training corpus. It is the first of two augmentation steps — the second (`02_request_framing.py`) replaces stub metadata with realistic HTTP headers and generates benign traffic.

The synthesis pipeline has three layers:

1. **Raw payload generation** — grammar templates and an optional LLM produce attack strings
2. **Transform chain** — every payload is obfuscated and encoded to produce WAF-evasion variants
3. **HTTP wrapping** — payloads are placed in a class-correct HTTP envelope for schema validation

---

## Architecture

```
taxonomy_inventory.json
        │
        ▼
AugmentationGovernor.gaps()          ← which classes need more samples?
        │
        ▼ (per class, parallel threads)
synthesize_class()
        │
        ├── GrammarGenerator          ← prefix + payload + suffix templates
        └── LlmGenerator (optional)   ← LLM-generated novel payloads
        │
        ▼ (every seed)
_build_chain()
        ├── ObfuscatorGenerator       ← class-specific syntax rewrite
        ├── ObfuscatorGenerator       ← stackable second pass (30%, lfi/ssrf only)
        ├── EncoderGenerator          ← byte-level encoding
        └── EncoderGenerator          ← double encoding (30%)
        │
        ▼
_payload_to_record()                  ← stub HTTP envelope + Pydantic validation
        │
        ▼
synthesized_attacks.parquet
```

---

## Generators

### GrammarGenerator

Produces raw payload strings by randomly sampling from per-class grammar templates:

```
payload = rng.choice(prefixes) + rng.choice(payloads) + rng.choice(suffixes)
```

Nine attack classes have grammars defined in `GRAMMAR_REGISTRY`:

| Class | Prefix examples | Payload examples | Suffix examples |
|---|---|---|---|
| `sqli` | `'`, `1 OR`, `" OR "1"="1` | `UNION SELECT NULL`, `AND SLEEP(5)`, `EXTRACTVALUE(...)` | `--`, `#`, `/**/` |
| `xss` | `<`, `"><`, `javascript:` | `script>alert(1)</script`, `img src=x onerror=...` | `>`, `/>` |
| `lfi` | `../`, `..\\`, `..%2f` | `../../etc/passwd`, `php://filter/...` | `%00`, `&` |
| `ssrf` | `http://`, `gopher://`, `file://` | `169.254.169.254/latest/meta-data/`, `127.0.0.1/admin` | `/`, `?debug=1` |
| `cmdi` | `; `, `\| `, `$(` | `id`, `cat /etc/passwd`, `bash -i >& /dev/tcp/...` | ` #`, ` 2>&1` |
| `path_traversal` | `%2e%2e/`, `..%5c` | `../../../../etc/shadow`, `....//etc/passwd` | `%00`, `.php` |
| `header_injection` | `%0d%0a`, `\r\n` | `Set-Cookie: session=evil`, `Location: https://evil.com` | `&` |
| `xxe` | `<?xml version="1.0"?>` | Full DOCTYPE+ENTITY documents for file read or SSRF | — |
| `ssti` | `{{`, `${`, `#{` | `{{7*7}}`, `{{request.application.__globals__...}}`, `<%= system('id') %>` | `}}`, `%}` |

### EncoderGenerator

Applies byte-level encoding transforms. Class-agnostic — the same encodings apply to all attack types since encoding operates at the character level, not the syntax level.

| Encoding | Example |
|---|---|
| `url_encode` | `' OR 1=1` → `%27%20OR%201%3D1` |
| `double_url_encode` | `'` → `%2527` |
| `partial_url_encode` | Randomly encodes ~50% of non-alphanumeric chars |
| `hex_encode` | Alpha chars → `0x41` literals |
| `unicode_escape` | Alpha chars → `A` |
| `html_entity` | Alpha chars → `&#65;` |
| `comment_insertion` | Spaces → `/**/` |
| `case_variation` | Random upper/lower per character |
| `whitespace_bypass` | Spaces → `\t`, `\n`, `%09`, `%0a` |

### ObfuscatorGenerator

Applies class-specific syntax rewrites that preserve the attack semantics while evading pattern-matching WAF rules.

**`sqli`**

| Transform | Effect |
|---|---|
| `apostrophe_mask` | `'` → `UTF8MB4_UNICODE_CI` |
| `modsecurity_safe` | `=` → ` LIKE `, `OR` → `\|\|` |
| `between` | `=1` → ` BETWEEN 0 AND 2` |
| `ifnull2ifisnull` | `IFNULL(` → `IF(ISNULL(` |
| `multiplespaces` | spaces → triple spaces |
| `space2dash` | spaces → `--\n` (SQL line comment) |
| `space2mssqlblank` | spaces → `\t` |

**`xss`**

| Transform | Effect |
|---|---|
| `tag_case` | `<script` → `<Script`, `<img` → `<Img` |
| `js_comment` | `alert(` → `alert/*xss*/(` |
| `backtick_exec` | `alert(1)` → `` alert`1` `` |
| `attr_double_encode` | `alert` → `&#x61;lert` |
| `null_byte_event` | `onerror=` → `on\x00error=` |

**`lfi`**

| Transform | Effect |
|---|---|
| `double_encode` | `../` → `%252e%252e%252f` |
| `overlong_utf8` | `../` → `%c0%ae%c0%ae/` |
| `dotdotslash` | `../` → `....//` |
| `null_byte` | appends `%00` |
| `backslash_mix` | `../` → `..\` |

**`cmdi`**

| Transform | Effect |
|---|---|
| `ifs_space` | spaces → `${IFS}` |
| `brace_expand` | `cat` → `{cat,}` |
| `quote_break` | `id` → `i''d` (empty string insertion) |
| `hex_cmd` | `cat` → `$'\x63\x61\x74'` |

**`ssrf`**

| Transform | Effect |
|---|---|
| `ip_decimal` | `127.0.0.1` → `2130706433` |
| `ip_octal` | `127.0.0.1` → `0177.0.0.1` |
| `ip_hex` | `127.0.0.1` → `0x7f000001` |
| `proto_confusion` | `http://` → `http:///` |
| `ipv6_mapped` | `127.0.0.1` → `[::ffff:127.0.0.1]` |

Classes without obfuscations defined (`xxe`, `ssti`, `header_injection`, `path_traversal`) skip the obfuscation step — only encoding is applied.

### LlmGenerator (optional)

Calls a configured LLM backend to generate novel payloads not derivable from the grammar templates. Uses class-specific prompts requesting JSON arrays of payload strings.

Supported backends:

| Provider | Auth | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Cloud API |
| `google` | `GOOGLE_API_KEY` | Cloud API |
| `ollama` | none | Local server at `ollama_base_url` |
| `local` | none | GGUF file via llama-cpp-python |

Disabled automatically if `LLM_PROVIDER` is unset or if `provider=local` and the GGUF file is not found.

---

## Transform Chain

Every seed payload — whether grammar-generated or LLM-generated — passes through `_build_chain` before being wrapped into an HTTP record:

```
seed
  │
  ▼ (always)
ObfuscatorGenerator  — one random class-specific syntax transform
  │
  ▼ (30% probability — lfi and ssrf only)
ObfuscatorGenerator  — safe stackable second transform:
                        lfi  → null_byte (appends %00, never conflicts)
                        ssrf → proto_confusion (rewrites scheme, safe after IP transforms)
  │
  ▼ (always)
EncoderGenerator     — one random byte-level encoding
  │
  ▼ (30% probability)
EncoderGenerator     — second encoding pass for double-encoded variants
  │
  ▼
final payload string
```

Both probabilities are configurable:

```yaml
augmentation:
  double_encode_prob:    0.30   # probability of second encoding pass
  double_obfuscate_prob: 0.30   # probability of stackable second obfuscation
```

---

## HTTP Wrapping

`_payload_to_record` places each transformed payload in a class-correct HTTP structure for Pydantic schema validation. The headers are intentionally minimal stubs — realistic metadata is applied by `02_request_framing.py`.

| Class | Method | Parameter | Endpoint |
|---|---|---|---|
| `sqli` | GET | `q` | `/api/v1/search` |
| `xss` | POST | `msg` | `/api/v1/comment` |
| `lfi` | GET | `path` | `/api/v1/file` |
| `ssrf` | GET | `url` | `/api/v1/fetch` |
| `cmdi` | POST | `host` | `/api/v1/ping` |
| `path_traversal` | GET | `file` | `/download` |
| `header_injection` | GET | `redirect` | `/api/v1/redirect` |
| `xxe` | POST | — | `/api/v1/xml` (raw body) |
| `ssti` | GET | `template` | `/api/v1/render` |

Records that fail Pydantic validation are silently dropped. The valid records are written to Parquet.

---

## Gap Analysis

`AugmentationGovernor` reads `reports/1_data_acquisition_and_curation/metrics/03_taxonomy_inventory.json` (produced by Stage 1) to determine how many samples each class currently has. Only classes below `target_per_class` are synthesised.

```
gap = max(0, target_per_class - current_count)
```

If the inventory file is missing, the full target is generated for all classes.

---

## Configuration

All parameters live in `config/pipeline.yaml` under `augmentation`:

```yaml
augmentation:
  target_per_class:      5000    # minimum samples per attack class
  chain_length:          2       # reserved, not currently used
  double_encode_prob:    0.30    # probability of second encoding pass
  double_obfuscate_prob: 0.30    # probability of stackable second obfuscation

  llm:
    provider:        "${LLM_PROVIDER}"    # anthropic | google | ollama | local
    model:           "mistral"
    model_path:      "${LOCAL_LLM_PATH}"  # required for provider=local
    ollama_base_url: "http://localhost:11434"
    max_tokens:      256
    temperature:     0.9
    request_timeout: 30

  rules:
    encodings: null   # list of EncoderGenerator encoding names to enable, or null for all
```

---

## Running

```bash
# Standard run (grammar + encoder + obfuscator only)
python stages/3_data_augmentation/01_attack_synthesis.py --config config/pipeline.yaml

# With local GGUF model
python stages/3_data_augmentation/01_attack_synthesis.py \
    --config config/pipeline.yaml \
    --model-path /path/to/model.gguf

# Force re-run even if output exists
python stages/3_data_augmentation/01_attack_synthesis.py --force

# Via make
make data_augment_all
```

Set `LLM_DEBUG=1` to log a truncated preview of every LLM response.

---

## Output

**Parquet file:** `data/augmented/synthesis/synthesized_attacks.parquet`  
Schema: `HttpRecord` — fields `id`, `method`, `path`, `query_string`, `headers`, `body`, `raw`, `label=1`, `attack_class`, `source`

**Stats file:** `reports/3_data_augmentation/metrics/01_augmentation_synthesis.json`

```json
{
  "target_per_class": 5000,
  "chain_length": 2,
  "double_encode_prob": 0.3,
  "double_obfuscate_prob": 0.3,
  "generators_used": ["grammar", "encoder", "obfuscator"],
  "llm_provider": "none",
  "llm_model": "none",
  "gaps_filled": {"sqli": 4823, "xss": 4991, ...},
  "total_generated": 43102
}
```

All metrics are also logged to MLflow under the run name `01_attack_synthesis`.

---

## What comes next

`02_request_framing.py` reads the synthesized Parquet, strips the stub headers, and re-wraps every attack record with realistic metadata sampled from `HttpMetadataDistribution` (varied user agents, auth headers, referers, hostnames, content types). It also generates benign traffic to balance the dataset.
