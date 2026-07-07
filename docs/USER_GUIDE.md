# User Guide

> **📖 Docs:** [Index](README.md) · [User Guide](USER_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [API](API.md) · [Stages](stages/stages.md) · [Model Card](MODEL_CARD.md) · [Repo README](../README.md)

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

Reports are organised **by pipeline stage** (mirroring `stages/`), keeping the
`` · `` · `figures/` split inside each stage and numbering files by
the substage that owns them:

```
reports/<N>_<stage>/<metrics|latency|figures>/<NN>_<name>
  e.g. reports/3_data_augmentation/metrics/04_quality_gate.json
       reports/7_evaluation/latency/02_latency_summary.json
       reports/7_evaluation/15_final_evaluation_report.json   (top-level deliverable)
```

`ai_waf_v2/utils/reports.py` (`REPORTS` / `report_path`) is the single source of
truth for these locations — writers and readers both resolve through it, so to move
the tree you edit that one file. Most JSON is **also logged to MLflow**, so you can
browse trends in the UI instead of opening files. The tables below list reports by
name in generation order; 📄 marks the ones you actually *read*.

If you only open four things: `corpus_report.json` (what data you have),
`quality_gate.json` + `token_leakage.json` (is the dataset clean and shortcut-free),
`master_comparison_table.csv` (how models stack up), and
`deployment_recommendation.json` (ship or not).

**Stage 1 — acquisition & curation** (`data_collect`, `data_analyze`)

| Report | Use it to… |
|---|---|
| `download_manifest.json` | confirm each dataset downloaded + SHA-256 verified |
| `collection_stats.json`, `normalization_stats.json` | sanity-check record counts after normalisation |
| `dedup_stats.json` | see how many cross-dataset duplicates were removed |
| `dataset_analysis.json`, `taxonomy_inventory.json`, `taxonomy_coverage.json`, `length_distribution.json`, `datasheet.json` | class×source coverage, **which classes are under-represented** (drives Stage 3), length percentiles, datasheet |
| 📄 `corpus_report.json` | the rolled-up "what's in my dataset" summary |

**Stage 2 — baselines** (`baselines`)

| Report | Use it to… |
|---|---|
| 📄 `slos.json` | the pass/fail bar for everything downstream (p99 < 5 ms inline / < 50 ms offline, FPR < 0.001) |
| `split_stats.json` | verify the 70/15/10/3/2 stratified split sizes |
| `baselines.json` | the **non-neural score to beat** (XGBoost / HashedNGram / Aho-Corasick) |
| `modsecurity_results.json` | signature-WAF reference (OWASP CRS PL1–PL4) with SLO verdicts |

**Stage 3 — augmentation** (`data_augment_all`)

| Report | Use it to… |
|---|---|
| `augmentation_synthesis.json` | per-class synthetic counts + generator origin (grammar/PCFG/LLM) |
| `traffic_distribution.json` | inspect the PCAP-fitted metadata used for realistic framing |
| `request_framing.json` | attack-reframe + benign-generation counts |
| 📄 `quality_gate.json` | dataset hygiene: `rejection_rate`, per-pass reasons, `n_quarantined` (manual review via `notebooks/03_quarantine_review.ipynb`), leakage removals |
| `augmentation_probe.json` | confirm synthetic data improves generalization |
| `taxonomy_inventory.json` (post-aug) | verify the Stage 1 class gaps are now closed |
| 📄 `token_leakage.json` | shortcut audit: `suspected_shortcuts`, `max_mi`, `counterfactual_swap.max_mi_drop` — confirms the model won't key on constants like `evil.com` |

**Stage 4 — tokenization** (`tokenize`)

| Report | Use it to… |
|---|---|
| `tokenizer_oov_track_a.json`, `tokenizer_oov_track_b.json` | OOV rate + token fertility per track |
| 📄 `tokenizer_comparison.json` | pick Track A vs B (OOV, fertility, vocab Jaccard, per-class OOV) |

**Stage 5–6 — training, distillation, export** (`train_b_99m`, `distill`)

| Report | Use it to… |
|---|---|
| `student_arch.json` | param breakdown + compression-ratio guard (teacher→student) |
| `student_threshold.json` | the calibrated temperature / decision threshold at the FPR target |
| `student_canary.json` | **memorization check** on the canary split |
| `*.json`, `onnx_bench.json`, `trt_bench.json` | export latency p50/p95/p99, throughput, VRAM — first SLO check |

**Stage 7 — evaluation** (`eval_only` / `compare_all` / `report`)

| Report | Use it to… |
|---|---|
| 📄 `detection_results.json` | primary efficacy: F1, AUC-PR/ROC, FPR, per-class, threshold@FPR=0.001 |
| `latency_summary.json`, `memory_footprint.json` | inline/offline latency + VRAM vs SLO |
| `adversarial_summary.json`, `obfuscation_robustness.json`, `novel_attack_generalization.json` | evasion, obfuscation, zero-shot robustness |
| `tokenizer_ablation.json`, `augmentation_ablation.json`, `model_size_scaling.json`, `label_smoothing_ablation.json` | which design choices actually mattered |
| `attention_visualization.json` (+ heatmaps), `shap_analysis.json`, `error_analysis.json` | interpretability / confusion-by-class debugging |
| `baselines_post_aug.json` | fair re-run of classical baselines on the augmented splits |
| 📄 `master_comparison_table.{json,csv}` | the one-look **all models × all metrics** table (7.14) |
| 📄 `reports/7_evaluation/15_final_evaluation_report.json` | the aggregated top-level report (7.15) |
| 📄 `deployment_recommendation.json` | automated go/no-go: model pick, SLO verdict, rollout plan (7.16) |

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

---

## 9. Task reference (all `run.py` targets)

Every task below is invoked as `python run.py <task>` and has an identical
`make <task>` equivalent. Composite tasks list their dependencies, which run
first in order. `python run.py --list` prints this list live; add `-n` /
`--dry-run` to any task to print the commands without running them, and pass
`--force` to an individual stage script to re-run it even if its outputs exist.

### Setup & quality

| Task | Description | Deps |
|---|---|---|
| `setup` | Create `.venv` and install deps (`MODE=full` \| `research`) | |
| `lint` | Ruff `check --fix` + format | |
| `typecheck` | Mypy on `stages/` | |
| `quality` | Lint + typecheck | `lint`, `typecheck` |
| `test` | Full test suite (`tests/test_all.py`) | |
| `test_core` | Model architecture tests (encoder + classifier) — `-k` subset of `test` | |
| `test_data` | Data schema + collator tests — `-k` subset of `test` | |
| `test_distill` | Distillation loss + metrics tests — `-k` subset of `test` | |

### Stage 1 — Acquisition & curation

| Task | Description |
|---|---|
| `data_collect` | Stage 1.1–1.2: acquire, normalize, cross-dataset dedup |
| `data_analyze` | Stage 1.3–1.4: corpus coverage report |

### Stage 2 — Baselines

| Task | Description |
|---|---|
| `baselines` | Stage 2: pre-augmentation baselines (split snapshot + ModSecurity CRS + classical ML) |
| `baselines_post_aug` | Stage 7: re-run classical baselines on the augmented splits |

### Stage 3 — Augmentation

| Task | Description | Deps |
|---|---|---|
| `data_augment_synthesis` | Stage 3.1: attack synthesis (grammar/PCFG + mutator + tamper + local LLM) | |
| `data_augment_benign` | Stage 3.2: traffic distribution profiler (PCAP alignment) | |
| `data_augment_framing` | Stage 3.3: request framing (reframe + benign REST + optional cloud LLM) | |
| `data_filter` | Stage 3.4: quality gate (format / UNK / class validity / dedup / label consistency) | |
| `data_validate` | Stage 3.5–3.6: augmentation probe + taxonomy inventory | |
| `data_split` | Stage 3.7: stratified split of the augmented corpus | |
| `data_leakage` | Stage 3.8: token-leakage probe (shortcut / filler-neutrality check) | |
| `data_augment_all` | Stage 3: full augmentation pipeline | `data_augment_synthesis`, `data_augment_benign`, `data_augment_framing`, `data_filter`, `data_validate`, `data_split`, `data_leakage` |

### Stage 4 — Tokenization

| Task | Description | Deps |
|---|---|---|
| `tokenize_a` | Stage 4A: augment a pretrained vocab + OOV report (Track A) | |
| `tokenize_b` | Stage 4B: custom BPE from scratch + OOV report (Track B) | |
| `tokenize_eval` | Stage 4: tokenizer comparison report | |
| `tokenize` | Stage 4: both tracks + comparison | `tokenize_a`, `tokenize_b`, `tokenize_eval` |

### Stage 5 — Teacher training

| Task | Description |
|---|---|
| `train_a_large` | Stage 5.1: Track A DeBERTa-v3-base fine-tune |
| `train_a_small` | Stage 5.2: Track A small pretrained fine-tune |
| `train_b_99m` | Stage 5.3: Track B 99M encoder from scratch |
| `distill_track_b` | Stage 5.3b: cross-track distillation (Track A teacher → Track B student) |

### Stage 6 — Distillation & compression

| Task | Description | Deps |
|---|---|---|
| `train_teacher` | Stage 6.0: train/resume the 99M teacher (re-entry point) | |
| `distill_train` | Stage 6.1–6.2: knowledge distillation → 10M student | |
| `distill_calibrate` | Stage 6.3: post-training temperature calibration | |
| `distill_canary` | Stage 6.4: student canary / memorisation check | |
| `distill_export` | Stage 6.5: ONNX + TorchScript export & latency bench | |
| `distill` | Stage 6: distill + calibrate + canary + export | `distill_train`, `distill_calibrate`, `distill_canary`, `distill_export` |

### Stage 7 — Evaluation

| Task | Description | Deps |
|---|---|---|
| `eval_detection` | Stage 7.1: detection efficacy | |
| `eval_latency` | Stage 7.2–7.3: latency/throughput + memory footprint | |
| `eval_adversarial` | Stage 7.4–7.6: adversarial robustness | |
| `eval_ablation` | Stage 7.7–7.10: ablation studies | |
| `eval_interp` | Stage 7.11–7.13: interpretability | |
| `compare_all` | Stage 7.14: master comparison table (runs all evals first) | |
| `report` | Stage 7.15: final report | |
| `deploy_reco` | Stage 7.16: deployment recommendation | |
| `push_hub` | Stage 7.17: push artifacts to HuggingFace Hub | |
| `push_hub_dry` | Stage 7.17: preview the Hub push (uploads nothing) | |

### Composite pipelines

| Task | Description | Deps |
|---|---|---|
| `all` | Full pipeline end-to-end | `setup`, `baselines`, `data_collect`, `data_analyze`, `data_augment_all`, `tokenize`, `train_b_99m`, `distill`, `compare_all`, `report` |
| `track_b_full` | Track B only — fastest path to results | `setup`, `baselines`, `data_collect`, `data_augment_all`, `tokenize_b`, `train_b_99m`, `distill`, `compare_all` |
| `eval_only` | Evaluate already-trained models | `eval_detection`, `eval_latency`, `eval_adversarial`, `eval_ablation`, `eval_interp`, `compare_all` |

### MLflow & housekeeping

| Task | Description | Deps |
|---|---|---|
| `ui` | Start the MLflow UI on `:5000` | |
| `stop_ui` | Stop the MLflow UI | |
| `clean_augmented` | Remove augmented parquet | |
| `clean_models` | Remove trained model weights | |
| `clean_tokenizers` | Remove tokenizer artifacts | |
| `clean_reports` | Remove report outputs | |
| `clean_splits` | Remove data splits | |
| `clean` | Remove all generated artifacts | `clean_augmented`, `clean_models`, `clean_tokenizers`, `clean_reports`, `clean_splits` |
