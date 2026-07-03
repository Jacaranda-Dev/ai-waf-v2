# ai-waf-v2

A Transformer-based **Web Application Firewall (WAF) classifier** trained with
knowledge distillation. It detects malicious HTTP requests (SQLi, XSS, LFI, SSRF,
CMDi, …) with a custom encoder trained from scratch on HTTP-aware tokenization,
then distills a 99M-parameter teacher into a ~10M student deployable at
single-digit-millisecond latency.

This repository is a reproducible, multi-stage ML research pipeline: every stage
from raw dataset acquisition through evaluation is a numbered, re-runnable script.

---

## Why

Signature-based WAFs (e.g. OWASP CRS) are brittle against obfuscation and
encoding mutations. ai-waf-v2 learns request structure directly, and is measured
head-to-head against classical baselines (XGBoost, ModSecurity CRS) on detection
efficacy, latency, and adversarial robustness.

Primary operating point: **FPR = 0.001** with a **p99 latency < 5 ms** inline SLO.

---

## Pipeline at a glance

```
 1. Data acquisition   public datasets ─► canonical HttpRecord ─► Parquet
 2. Baselines          XGBoost · ModSecurity CRS · SLO definitions
 3. Augmentation       grammar/mutator/tamper/LLM synthesis ─► quality gate ─► split
 4. Tokenization       Track A (augmented BERT vocab)  vs  Track B (custom BPE)
 5. Teacher training    99M WafEncoder from scratch  (+ DeBERTa Track A variants)
 6. Distillation        99M ─► ~10M student · calibrate · canary · ONNX export
 7. Evaluation          detection · latency · adversarial · ablations · report
```

Each stage consumes the previous stage's artefacts; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data-flow diagram.

---

## Quick start

```bash
# 1. Prerequisites: Python 3.10+, git. (Optional: a CUDA GPU for training.)
git clone <repo-url> ai-waf-v2 && cd ai-waf-v2

# 2. A .env file is required (it may be empty). It holds API keys / config:
touch .env

# 3. Create the venv and install dependencies
python run.py setup           # cross-platform, no `make` needed
#   ...or, on Linux/macOS with make installed:
make setup

# 4. Run the fastest end-to-end path (Track B only)
python run.py track_b_full
```

> **No `make`?** Every `make <target>` has an identical
> `python run.py <target>` (pure standard library, works on Windows/macOS/Linux).
> Run `python run.py --list` to see all tasks, or `-n` to dry-run.

See the [User Guide](docs/USER_GUIDE.md) for the full walkthrough.

---

## Documentation

**📖 Start at the [documentation index](docs/README.md)** — every doc links back to
it and to its neighbours.

| Document | What it covers |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index / navigation hub |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Setup → run the pipeline → read results; every command and env knob |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline DAG, core library design, data flow, config system |
| [docs/API.md](docs/API.md) | `ai_waf_v2` library reference (public API) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, training data, metrics, limitations |
| [docs/stages/](docs/stages/) | Deep dive per pipeline stage |
| [docs/attacks/](docs/attacks/) | Attack taxonomy & synthesis notes (SQLi, XSS, CMDi, …) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev workflow, quality checks, conventions |
| [CHANGELOG.md](CHANGELOG.md) | Notable changes |

For agent-assisted development, [CLAUDE.md](CLAUDE.md) documents the codebase for
Claude Code.

---

## Common commands

```bash
python run.py --list          # all tasks
python run.py quality         # ruff lint/format + mypy
python run.py test            # full test suite (synthetic data, no GPU/network)
python run.py data_collect    # Stage 1
python run.py train_b_99m     # Stage 5 teacher
python run.py distill         # Stage 6 student
python run.py eval_only       # Stage 7 on existing checkpoints
python run.py ui              # MLflow UI at localhost:5000
```

## License

See repository metadata.
