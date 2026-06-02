# Stage 6 — Distillation & Compression

**Directory:** `stages/6_distillation_and_compression/`  
**Make target:** `make distill_train`  
**Outputs:** `models/student/best_student.pt`, `models/student/student.onnx`, `reports/latency/latency_summary.json`

---

## Purpose

Stage 6 compresses the 99M Track B teacher from Stage 5 into a deployable 10M student via knowledge distillation, then verifies and exports it. This is the second of two distillation steps in the pipeline — see [Stage 5 doc](stage5_teacher_training.md) for why both exist.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Compression for deployment                               │
│                                                                     │
│  Track B 99M  ──[frozen teacher]──▶  Student 10M                   │
│      99M                                  10M                      │
│   BPE vocab                            BPE vocab                   │
│                                                                     │
│  Both models share the same HttpTokenizer vocabulary.              │
│  Output: INT8-quantized, ONNX-exported, SLO-verified model.        │
└─────────────────────────────────────────────────────────────────────┘
```

Concretely: teacher and student share the Track B BPE vocabulary (8k tokens), so the teacher's logit distributions are directly comparable. The student is ~10× smaller (d_model=256, 6 layers vs 768/13), quantized to INT8, and exported to ONNX and TensorRT for deployment.

---

## Script overview

| Script | Purpose | Key output |
|---|---|---|
| `00_train_teacher_99m.py` | Ensure Track B 99M teacher checkpoint exists | `models/track_b/99m/best_99m.pt` |
| `01_student_arch.py` | Pre-flight: validate dimensions, log compression ratio | `reports/metrics/student_arch.json` |
| `02_distill_train.py` | Core KD training: 99M → 10M student with INT8 QAT | `models/student/best_student.pt` |
| `03_student_calibrate.py` | Sweep thresholds on val set; find FPR-constrained optimum | `reports/metrics/student_threshold.json` |
| `04_student_canary.py` | Regression gate: per-family recall on synthetic payloads | `reports/metrics/student_canary.json` |
| `05_export_and_bench.py` | ONNX + TRT export; SLO-enforced latency benchmark | `reports/latency/latency_summary.json` |

---

## Script details

### 6.0 — `00_train_teacher_99m.py`

Re-entry point that trains (or resumes) the Track B 99M teacher if `best_99m.pt` is missing. Downstream scripts all require this checkpoint; running `00` first makes the rest of Stage 6 self-contained.

Key checks before training starts:
- **Vocab consistency**: `teacher_cfg.vocab_size` must match the loaded `HttpTokenizer.vocab_size`. A mismatch would cause the teacher's soft-target distributions to misalign with the student's embeddings.
- **Parameter count guard**: warns if the instantiated model is outside 90M–110M parameters (catches misconfigured dimensions early).

Training uses `OneCycleLR` with a 5% warmup phase, bf16 autocast, and `AdamW` with fused CUDA kernels when available. Saves `checkpoint_epoch{N:03d}.pt` each epoch; `best_99m.pt` tracks the lowest validation loss; `latest_checkpoint.pt` is a symlink for easy resume. Uses `WafClassifier.from_config` from `ai_waf_v2.models.head`.

---

### 6.1 — `01_student_arch.py`

Architecture pre-flight check. Runs entirely on CPU with no data loading.

Validates:
- `student.vocab_size == teacher.vocab_size` — raises `ValueError` on mismatch (would produce invalid soft targets).
- Compression ratio is between 5× and 50×: below 5× the student is too large for meaningful latency gains; above 50× the accuracy drop is likely unacceptable. Both conditions emit warnings, not errors.

Logs a formatted parameter breakdown (embeddings / attention / FFN / head) and VRAM estimates at fp16 and INT8:

```
Student Architecture Summary
============================
  d_model       : 256
  n_layers      : 6
  n_heads       : 4
  vocab_size    : 8000
  quantization  : int8
  ─────────────────────
  Embeddings    :    2,048,000
  Attention     :    2,359,296
  FFN           :    5,505,024
  Head          :      524,290
  Total params  :   10,436,610
  ─────────────────────
  Compression   :   9.5× vs teacher 99M
  VRAM fp16     :  20 MB
  VRAM int8     :  10 MB
```

---

### 6.2 — `02_distill_train.py`

Core distillation. The teacher is loaded frozen; only student weights are updated.

#### Loss function

```
L = α_hard × CE(student_logits, hard_labels)
  + α_soft × T² × KL( log_softmax(student/T) ‖ softmax(teacher/T) )
  + w_mse  × MSE(proj(student_hidden), teacher_hidden)
```

| Term | Default weight | Purpose |
|---|---|---|
| CE hard labels | α_hard = 0.3 | Ground-truth supervision |
| T²·KL soft labels | α_soft = 0.7 | Teacher's uncertainty/inter-class signal |
| Hidden MSE | w_mse = 0.0 | CLS-embedding alignment (optional) |

The T² factor rescales the KL gradient to match CE magnitude regardless of temperature. `alpha_hard + alpha_soft` is validated to equal 1.0 at construction.

#### Temperature annealing

Instead of a fixed temperature, a `TemperatureScheduler` linearly decays T from `T_max` → `T_min` over training:

- **Early epochs (high T):** teacher distribution is softened → student learns broad inter-class structure ("dark knowledge").
- **Late epochs (low T):** distribution sharpens → student locks in fine-grained discrimination.

Defaults: `T_max=6.0`, `T_min=1.5`. Both are configurable via `config.training.distillation.temperature_max/min`.

#### INT8 quantization-aware training (QAT)

When `config.model.student.quantization = "int8"`, the student's encoder `nn.Linear` layers are replaced with `bitsandbytes.nn.Linear8bitLt` before training starts. The classification **head is excluded** because SiLU/GELU activations in the head lose calibration accuracy under INT8 precision. Any head modules using precision-sensitive activations are detected automatically and logged.

If `bitsandbytes` is not installed, QAT is skipped with a warning and training continues in fp32.

The actual training loop is delegated to `ai_waf_v2.distill.trainer.DistillationTrainer`, which handles: AdamW with no-decay for bias/LayerNorm, linear warmup + cosine decay LR, bf16 mixed precision, gradient accumulation, early stopping on val AUC-PR.

---

### 6.3 — `03_student_calibrate.py`

Distillation shifts the student's logit scale relative to the teacher. Deploying with the teacher's threshold (or naively 0.5) inflates the False Positive Rate. This script finds the optimal threshold for the student specifically.

Method:
1. Run student inference over the full validation set; collect `softmax(logits)[:, 1]` (attack probability).
2. Sweep 400 evenly-spaced thresholds from `min_score` to `max_score`.
3. At each threshold compute precision, recall, F1, FPR.
4. Select the threshold with **highest F1 subject to FPR ≤ `cfg.slo.max_fpr`** (default 0.005).
5. If no threshold satisfies the FPR constraint, fall back to the threshold with the lowest FPR overall and emit a warning.

Output `student_threshold.json`:
```json
{
  "calibrated_threshold": 0.4312,
  "metrics_at_threshold": {"f1": 0.9963, "precision": 0.9971, "recall": 0.9956, "fpr": 0.0004},
  "slo_max_fpr": 0.005,
  "slo_satisfied": true,
  "full_sweep": [...]
}
```

Script 6.4 reads `calibrated_threshold` from this file; 6.5 checks `slo_satisfied`.

---

### 6.4 — `04_student_canary.py`

Regression gate. Distillation can cause the student to forget rare attack patterns that relied on the teacher's larger capacity — particularly long-range attention over obfuscated injections. This script catches that before export.

14 built-in synthetic payloads cover all major attack families:

| Family | Example |
|---|---|
| `sqli_classic` | `GET /search?q=1' OR '1'='1` |
| `sqli_union` | `UNION SELECT null,username,password FROM users--` |
| `sqli_obfuscated` | `/*!50000OR*/1=1--` |
| `xss_basic` | `<script>alert(1)</script>` |
| `xss_encoded` | `%3Cscript%3Ealert%281%29%3C%2Fscript%3E` |
| `xss_event` | `<img src=x onerror=alert(1)>` |
| `path_traversal` | `../../../../etc/passwd` |
| `path_traversal_encoded` | `..%2F..%2F..%2Fetc%2Fshadow` |
| `cmdi` | `127.0.0.1;cat+/etc/passwd` |
| `cmdi_pipe` | `cmd=ls\|whoami` |
| `ssrf` | `http://169.254.169.254/latest/meta-data/` |
| `ssrf_encoded` | `http%3A%2F%2F169.254.169.254%2F` |
| `header_inject` | `X-Forwarded-For: 127.0.0.1 / Host: evil.com` |
| `long_obfuscated` | 12× repeated URL-encoded SQL (stress-tests long-range attention) |

Alternatively, pass `--canary-dir <path>` to load `.jsonl` files (each keyed as a family by filename stem).

For each family, recall = fraction of payloads detected at the calibrated threshold. If any family's recall drops below `cfg.slo.canary_min_recall` (default 0.90), the script **exits with code 1** — CI/CD can use this to block deployment of bad checkpoints.

Script 6.5 reads `student_canary.json` and refuses to export if `slo_passed` is false.

---

### 6.5 — `05_export_and_bench.py`

Runs the full export and benchmark chain in one pass, enforcing an end-to-end latency SLO across all inference backends.

Pipeline:

```
1. Check canary gate (refuse export if 04 failed)
2. Benchmark PyTorch baseline (bf16, n_warmup=50, n_runs=200)
3. Export to ONNX
     - opset 17
     - do_constant_folding=True
     - dynamic axes: batch_size and seq_len for both inputs
     - graph validated with onnx.checker (if onnx installed)
4. Benchmark ORT — CUDAExecutionProvider
5. Benchmark ORT — CPUExecutionProvider
6. Build TensorRT FP16 engine (skipped gracefully if tensorrt absent)
7. Benchmark TRT engine
8. SLO enforcement across all providers
9. Write latency_summary.json + deployment_status.json
```

SLO checks per provider:
- **p99 at bs=1** ≤ `cfg.slo.latency_inline_p99_ms` (default 5 ms)
- **Peak throughput** ≥ `cfg.slo.throughput_min_rps`

A provider is `SKIPPED` (not `FAIL`) if it was not built/available. `APPROVED` requires all non-skipped providers to `PASS`. If any provider fails, the script **exits with code 1** and prints which SLO was violated.

Artifacts written:

| File | Content |
|---|---|
| `models/student/student.onnx` | ONNX model, opset 17, dynamic axes |
| `models/student/student.engine` | TensorRT FP16 engine (if TRT available) |
| `reports/latency/latency_summary.json` | Unified p50/p95/p99/throughput across all providers |
| `reports/latency/onnx_bench.json` | PyTorch + ORT-CUDA results |
| `reports/latency/ort_detailed_bench.json` | Per-provider ORT breakdown |
| `reports/latency/trt_bench.json` | TRT results (if engine built) |

---

## Shared infrastructure

Stage 6 uses the canonical `ai_waf_v2` library directly (no local `train_utils` / `checkpoint_utils` shim layer like Stage 5):

| Module | Used by |
|---|---|
| `ai_waf_v2.models.head.WafClassifier` | 00 — teacher model |
| `ai_waf_v2.models.student.StudentClassifier` | 02, 03, 04, 05 |
| `ai_waf_v2.distill.losses.DistillationLoss` | 02 (via `DistillationTrainer`) |
| `ai_waf_v2.distill.trainer.DistillationTrainer` | 02 |
| `ai_waf_v2.eval.latency.LatencyBenchmark` | 05 |
| `ai_waf_v2.eval.latency.OnnxLatencyBenchmark` | 05 |
| `ai_waf_v2.tokenizer.http_tokenizer.HttpTokenizer` | 00, 02, 03, 04 |
| `ai_waf_v2.data.dataset.WafDataset` | 00, 02, 03 |
| `ai_waf_v2.data.collator.WafCollator` | 00, 02, 03 |

---

## Configuration reference

All values live in `config/pipeline.yaml`.

| Key | Default | Used by |
|---|---|---|
| `model.track_b_99m.output_dir` | `models/track_b/99m` | 00, 02 |
| `model.track_b_99m.vocab_size` | 8000 | 00, 01 |
| `model.track_b_99m.d_model` | 768 | 01, 02 |
| `model.student.output_dir` | `models/student` | 02, 03, 04, 05 |
| `model.student.d_model` | 256 | 01, 02 |
| `model.student.n_layers` | 6 | 01 |
| `model.student.n_heads` | 4 | 01 |
| `model.student.vocab_size` | 8000 | 01 |
| `model.student.quantization` | `int8` | 02 |
| `training.distillation.epochs` | — | 02 |
| `training.distillation.temperature` | 4.0 | 02 |
| `training.distillation.temperature_max` | 6.0 | 02 |
| `training.distillation.temperature_min` | 1.5 | 02 |
| `training.distillation.alpha_soft` | 0.7 | 02 |
| `training.distillation.alpha_hard` | 0.3 | 02 |
| `training.distillation.hidden_mse_weight` | 0.0 | 02 |
| `training.distillation.batch_size` | — | 02 |
| `training.teacher.epochs` | — | 00 |
| `training.teacher.lr` | — | 00 |
| `training.teacher.weight_decay` | — | 00 |
| `training.teacher.grad_clip` | — | 00 |
| `slo.max_fpr` | 0.005 | 03 |
| `slo.canary_min_recall` | 0.90 | 04 |
| `slo.latency_inline_p99_ms` | 5.0 | 05 |
| `slo.throughput_min_rps` | — | 05 |
| `evaluation.batch_sizes` | `[1, 8, 64]` | 05 |
| `evaluation.n_latency_warmup` | — | 05 |
| `evaluation.n_latency_runs` | — | 05 |
| `tokenizer.track_b.output_dir` | `tokenizers/track_b` | 00, 02, 03, 04 |
| `tokenizer.seq_len` | 256 | all |

---

## Running

```bash
# Full Stage 6 via make
make distill_train

# Individual scripts (must run in order)
cd stages/6_distillation_and_compression

python 00_train_teacher_99m.py --config config/pipeline.yaml
python 01_student_arch.py       --config config/pipeline.yaml
python 02_distill_train.py      --config config/pipeline.yaml
python 03_student_calibrate.py  --config config/pipeline.yaml
python 04_student_canary.py     --config config/pipeline.yaml
python 05_export_and_bench.py   --config config/pipeline.yaml

# Force re-run of a specific step
python 02_distill_train.py --config config/pipeline.yaml --force

# Resume teacher training from a checkpoint
python 00_train_teacher_99m.py --config config/pipeline.yaml \
    --resume models/track_b/99m/latest_checkpoint.pt

# Use a non-default teacher checkpoint for distillation
python 02_distill_train.py --config config/pipeline.yaml \
    --teacher-path models/track_b/99m/checkpoint_epoch005.pt

# Use external canary payloads (JSONL files, one family per file)
python 04_student_canary.py --config config/pipeline.yaml \
    --canary-dir data/canary_payloads/
```

### Dependencies for optional features

| Feature | Required package |
|---|---|
| INT8 QAT | `bitsandbytes` |
| ONNX graph validation | `onnx` |
| TensorRT engine build | `tensorrt`, `pycuda` |
| ORT GPU inference | `onnxruntime-gpu` |

Install all at once: `make setup MODE=full`

---

## Data flow

```
Stage 5 → models/track_b/99m/latest/  (teacher checkpoint)
Stage 3 → data/splits/train.parquet, val.parquet
Stage 4 → tokenizers/track_b/

  00_train_teacher_99m  →  models/track_b/99m/best_99m.pt
  01_student_arch       →  reports/metrics/student_arch.json
  02_distill_train      →  models/student/best_student.pt
  03_student_calibrate  →  reports/metrics/student_threshold.json
  04_student_canary     →  reports/metrics/student_canary.json
  05_export_and_bench   →  models/student/student.onnx
                           models/student/student.engine  (TRT, if available)
                           reports/latency/latency_summary.json

→ Stage 7: all evaluation scripts load from models/student/
```

---

## Deployment gates

Stage 6 has three hard gates that stop the pipeline with `exit(1)`:

1. **Canary gate** (`04`): any attack family below recall SLO → blocks export in `05`.
2. **FPR gate** (`03`): warns if calibrated threshold cannot satisfy FPR SLO; `05` checks `slo_satisfied`.
3. **Latency SLO** (`05`): any non-skipped inference backend exceeding p99 or throughput targets → blocks deployment approval.

All three results are written as JSON so CI/CD can read them independently of the exit code.
