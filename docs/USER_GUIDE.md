# User Guide

This guide walks from a fresh clone to a trained, evaluated model. It assumes no
prior knowledge of the codebase. For the design rationale behind the pipeline,
see [ARCHITECTURE.md](ARCHITECTURE.md); for the library API, see [API.md](API.md).

> **`make` vs `run.py`** — Every command below is shown as `python run.py <task>`,
> which works on Windows, macOS, and Linux with no extra tooling. If you have
> `make`, `make <task>` is equivalent. `python run.py --list` shows every task;
> add `-n` to print the underlying commands without running them.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Checked by `pyproject.toml` |
| git | For cloning |
| A CUDA GPU | **Optional.** Needed only to *train* in reasonable time. Tests, data, and tokenizer stages run CPU-only. |
| `.env` file | **Required for every task** (may be empty) — see below |

### The `.env` file

A `.env` file in the repo root is mandatory. `run.py` loads it automatically and
the `Makefile` includes it; a missing file makes `make` hard-fail. Create it once:

```bash
touch .env
```

Populate it only with what you need:

```ini
# LLM-based augmentation (Stage 3) — optional
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
# HuggingFace publishing (Stage 7.17) — optional
HF_TOKEN=hf_...
HF_ORG=your-org
# Remote MLflow (otherwise runs are stored locally) — optional
MLFLOW_TRACKING_URI=...
```

Real shell environment variables take precedence over `.env`.

---

## 2. Install

```bash
python run.py setup            # MODE defaults to "full"
```

`setup` creates `.venv/`, installs the package editable, and registers a Jupyter
kernel. Dependency extras are selected by `MODE`:

| `MODE` | Installs | Use when |
|---|---|---|
| `full` (default) | core + ONNX, TensorRT, bitsandbytes | Training, quantization, export |
| `research` | core + Jupyter only | Notebooks / data exploration, no GPU export |

```bash
MODE=research python run.py setup     # lighter install
```

After setup, prefer the venv interpreter. `run.py` auto-detects `.venv` and uses
it; if none exists it falls back to the current interpreter and warns.

---

## 3. Verify the install

```bash
python run.py quality     # ruff lint/format + mypy on stages/
python run.py test        # full suite — synthetic in-memory data, no GPU/network
```

Scoped test subsets:

```bash
python run.py test_core      # encoder + classifier
python run.py test_data      # schema + collator
python run.py test_distill   # distillation loss + metrics
# or directly:
.venv/bin/python -m pytest tests/test_all.py -k WafEncoder -v
```

---

## 4. Run the pipeline

Stages are sequential — each depends on the previous stage's outputs. Each script
is **idempotent**: it checks for its own outputs and skips unless you pass
`--force`. You can run a whole stage, or an individual sub-step.

### The fast path

To go from nothing to a compared model on the custom-tokenizer track:

```bash
python run.py track_b_full
# = setup → baselines → data_collect → data_augment_all
#   → tokenize_b → train_b_99m → distill → compare_all
```

The full end-to-end pipeline (both tokenizer tracks, all teachers):

```bash
python run.py all
```

### Stage by stage

| Stage | Task | Produces |
|---|---|---|
| 1 — Acquire & curate | `data_collect`, `data_analyze` | `data/normalized/*.parquet`, corpus report |
| 2 — Baselines | `baselines` | `data/splits/`, `reports/slos.json`, baseline metrics |
| 3 — Augmentation | `data_augment_all` | augmented + re-split `data/splits/` |
| 4 — Tokenization | `tokenize_b` (or `tokenize` for both) | `tokenizers/track_b/` |
| 5 — Teacher | `train_b_99m` | `models/track_b/99m/best_99m.pt` |
| 6 — Distill | `distill` | `models/student/` (+ calibrated, ONNX) |
| 7 — Evaluate | `eval_only`, `report` | `reports/` tables, figures, final report |

Run a single sub-step by its task name, e.g.:

```bash
python run.py data_filter        # just Stage 3.4 quality gate
python run.py distill_export     # just the ONNX/TorchScript export + bench
python run.py --list             # discover every task
```

### Stage 3 augmentation knobs (env vars)

Stage 3 reads optional environment variables to enable LLM/PCAP features:

| Variable | Effect |
|---|---|
| `LOCAL_LLM_PATH` | Path to a local GGUF model for offline attack synthesis |
| `LLM_PROVIDER` | `anthropic` \| `google` \| `local` \| `ollama` for benign generation |
| `PCAP_DIR` | Directory of PCAP traces to fit realistic traffic distributions |
| `HF_SAVE`, `HF_REVISION` | Save synthesised datasets to HuggingFace |

```bash
LLM_PROVIDER=anthropic python run.py data_augment_framing
```

---

## 5. Inspect results

### MLflow

All training runs are logged to MLflow. Browse them:

```bash
python run.py ui            # http://localhost:5000
python run.py stop_ui
```

Runs are local (SQLite) unless `MLFLOW_TRACKING_URI` is set.

### Reports

Stage 7 writes to `reports/`:

- `reports/metrics/` — detection, latency, baseline JSON
- `reports/comparison_table.{md,tex}` — all models × all metrics
- `reports/eval_report.{json,html}` — aggregated evaluation
- `reports/deployment_recommendation.md` — `python run.py deploy_reco`

---

## 6. Publish (optional)

Push the student model, ONNX weights, and report to the HuggingFace Hub:

```bash
export HF_TOKEN=hf_... HF_ORG=your-org
python run.py push_hub_dry      # preview — uploads nothing
python run.py push_hub          # upload
```

Scope a push with `HF_ONLY` and choose a branch with `HF_REVISION`.

---

## 7. Re-running and cleaning

Force a stage to re-run even if outputs exist:

```bash
.venv/bin/python stages/3_data_augmentation/04_quality_gate.py --config config/pipeline.yaml --force
```

Granular cleanup (none of these touch raw datasets):

```bash
python run.py clean_models       # trained weights
python run.py clean_tokenizers
python run.py clean_reports
python run.py clean_splits
python run.py clean              # all of the above
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `make: *** No rule to make target '.env'` | Missing `.env`. Run `touch .env`. |
| `run.py` warns "no .venv found" | You haven't run `setup`, or aren't in the repo root. |
| CUDA / TensorRT import errors | Installed with `MODE=research`; reinstall with `MODE=full`, or skip GPU-only stages. |
| A stage "does nothing" | It detected existing outputs and skipped — pass `--force` to the script. |
| LLM augmentation skipped | No API key in `.env` / no `LLM_PROVIDER`; the stage falls back to non-LLM generators. |

All hyperparameters live in [`config/pipeline.yaml`](../config/pipeline.yaml) —
there are no magic numbers in code. Edit there to change splits, model sizes,
SLOs, or augmentation behaviour.
