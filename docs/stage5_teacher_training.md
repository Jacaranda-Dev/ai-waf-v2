# Stage 5 — Teacher Training

**Directory:** `stages/5_teacher_training/`  
**Make target:** `make train_b_99m` (Track B primary path); Track A scripts run individually  
**Outputs:** `models/track_a/large/latest`, `models/track_a/small/latest`, `models/track_b/99m/latest`

---

## Overview

Stage 5 produces the **teacher models** that Stage 6 distils into the deployable 10M student. Three separate training runs are defined, each producing a checkpoint under `models/`:

| Script | Model | Backbone | ~Params | Config key |
|---|---|---|---|---|
| `01_track_a_large.py` | DeBERTa-v3-base (Track A large) | Pretrained, fine-tuned | 337M | `model.track_a_large` |
| `02_track_a_small.py` | Google BERT-tiny (Track A small) | Pretrained, fine-tuned | ~4M | `model.track_a_small` |
| `03_track_b_99m.py` | Custom Transformer (Track B, supervised) | From scratch | ~99M | `model.track_b_99m` |
| `03b_distill_track_b.py` | Track B 99M (distilled) | Warm-started from 03 | ~99M | `model.track_b_99m` |

Scripts 01 and 02 are independent. Script 03b depends on both 01 (teacher weights) and 03 (student warm-start checkpoint).

---

## Two distillation steps — why both exist

There are two distillation steps in the pipeline and they serve completely different purposes:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — 03b_distill_track_b.py   (quality transfer)             │
│                                                                     │
│  DeBERTa-v3-base  ──[frozen teacher]──▶  Track B 99M               │
│       337M                                   99M                   │
│   WordPiece vocab                         BPE vocab                │
│                                                                     │
│  Goal: teach the custom-tokenizer model to mimic DeBERTa's         │
│        decision boundaries despite using a different tokenizer.     │
│        Output is still a large (~99M) model — NOT deployable.      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  models/track_b/99m/latest
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — distill_train   (compression for deployment)            │
│                                                                     │
│  Track B 99M  ──[frozen teacher]──▶  Student 10M                   │
│      99M                                 10M                       │
│   BPE vocab                           BPE vocab                   │
│                                                                     │
│  Goal: compress the teacher 10× into a model that meets the        │
│        SLO (p99 latency <5 ms inline). Output is deployable.       │
└─────────────────────────────────────────────────────────────────────┘
```

**Stage 5 `03b`** is a *quality transfer*: DeBERTa-v3 has strong representations from pretraining on 160 GB of text; the Track B model was trained from scratch on HTTP data only. Distillation bridges the gap — the 99M student learns to match DeBERTa's soft output distribution even though it uses a completely different tokenizer. The two models never share token IDs; they share only the final logit distribution over `{benign, malicious}`.

**Stage 6** is a *compression*: the 99M teacher (supervised or distilled, whichever `models/track_b/99m/latest` points to) is reduced 10× to a ~10M student. Same tokenizer on both sides. This is the step that makes the model deployable within latency SLOs.

`03b` is **optional**. If skipped, Stage 6 distils from the supervised 03 checkpoint. Running `03b` typically improves the final student quality at the cost of the extra training run.

---

## Shared Infrastructure

Both training primitives and checkpoint management live in dedicated modules so that no script duplicates boilerplate.

### `train_utils.py`

Provides the full training primitive set imported by all four scripts.

**`WafClassifier`** — track-agnostic classification head. Accepts `last_hidden_state` from any backbone (HuggingFace AutoModel or `TinyTransformerEncoder`), applies `[CLS]` pooling, LayerNorm, Dropout, and a linear projection to `num_labels`. Identical across all three tracks for a fair comparison.

**`WafCollator`** — padding-aware batch builder compatible with both HuggingFace tokenizers (Track A, exposes `pad_token_id`) and `HttpTokenizer` (Track B, exposes `pad_id`). Pads each batch to its own longest sequence, not a global `max_length`.

**`build_optimizer(model, lr, weight_decay, warmup_steps, total_steps, backbone_lr_multiplier)`** — AdamW with:
- Differential LRs: parameters whose names contain `classifier` or `linear` use `lr`; all others use `lr * backbone_lr_multiplier`. For fine-tuning (scripts 01/02) `backbone_lr_multiplier=0.1` or `0.3`, keeping the pretrained backbone conservative. For training from scratch (script 03/03b) it is `1.0`.
- Linear warmup followed by linear decay to zero. The scheduler is stepped after every accumulation window (not every micro-batch) so the LR schedule matches effective batch steps.
- No weight decay on bias or LayerNorm parameters (handled implicitly by AdamW's default `weight_decay` applied only to the groups returned by the loop).

**`run_epoch(model, loader, optimizer, scheduler, device, accum_steps, scaler, loss_fn)`** — single training epoch with:
- Gradient accumulation over `accum_steps` micro-batches.
- **Mixed precision**: when `scaler` is provided (a `torch.amp.GradScaler`), uses fp16 autocast; without a scaler, uses bf16. This distinction matters: DeBERTa-v3 has fp16 checkpoint weights that produce fp16 gradients under bf16 autocast, causing GradScaler's `unscale_()` to fail. The fix is to pass a scaler (scripts 01) and call `.float()` on the model after `.to(device)` so all parameters are fp32 before training begins.
- Gradient clipping to norm 1.0 after `scaler.unscale_()` (or directly for the bf16 path).
- **Non-finite loss guard**: skips the backward and resets accumulated gradients when `loss.item()` is NaN or Inf. If 20 consecutive batches are non-finite (`_NAN_STREAK_LIMIT`), raises `RuntimeError` — the model weights have gone NaN and there is no recovery within the epoch.
- The running loss displayed in the progress bar is `total_loss / (i + 1)` — a true running average, not a double-counted estimate.

**`evaluate(model, loader, device, num_labels)`** — full-dataset evaluation returning `loss`, `accuracy`, `macro_f1`, and `per_class_f1`. Uses sklearn's `f1_score` with `zero_division=0`. Runs under the same autocast context as training.

**`flatten_metrics(metrics, prefix)`** — flattens nested dicts (e.g. `per_class_f1: {0: 0.97, 1: 0.94}`) into dot-notation keys for MLflow logging.

---

### `checkpoint_utils.py`

Unified checkpoint management so Stage 6 can always reference a static symlink path regardless of training timestamp or step count.

**`save_checkpoint(model, meta, ckpt_dir)`** — writes `model_weights.pt` and `checkpoint_meta.json` to a temporary sibling directory, then atomically renames it into place. A failed write never leaves a corrupt checkpoint.

**`load_checkpoint(path, model, device)`** — restores weights in-place. Accepts a checkpoint directory (looks for `model_weights.pt` inside) or a direct `.pt` path. Resolves symlinks before reading.

**`CheckpointTracker`** — wraps the best-metric logic:
- Saves a checkpoint at every evaluation step under `models/<track>/<experiment>/step_<NNNNNN>/`.
- Updates the `latest` symlink only when a new best is found.
- Increments a `no_improve` counter; sets `should_stop = True` when it reaches `patience`.
- Tracked metric and mode (`max`/`min`) are configurable per script.

**`resolve_checkpoint(experiment_type, cfg)`** — used by `03b_distill_track_b.py` and Stage 6 scripts to locate a checkpoint. Checks `models/<track>/latest` first, then falls back to an explicit `checkpoint_dir` in config. Raises `FileNotFoundError` with a clear message if neither exists.

Checkpoint layout:
```
models/
  track_a/
    large/
      latest  →  step_000024/   (symlink to best epoch)
      step_000001/
        model_weights.pt
        checkpoint_meta.json
      step_000024/
        …
  track_b/
    99m/
      latest  →  step_049000/
      …
```

---

## Scripts

### `01_track_a_large.py` — DeBERTa-v3-base fine-tuning

**Inputs:** `data/splits/train.parquet`, `data/splits/val.parquet`  
**Outputs:** `models/track_a/large/latest` (best epoch by macro-F1)

Fine-tunes `microsoft/deberta-v3-base` (337M parameters) on the WAF binary classification task. The backbone is loaded in fp32 and then `.float()` is applied after `.to(device)` — DeBERTa-v3's checkpoint contains some fp16 weights which would otherwise create fp16 gradients at the first accumulation step, causing `GradScaler.unscale_()` to raise `ValueError: Attempting to unscale FP16 gradients`.

**AMP setup** (DeBERTa-specific): fp16 autocast + `torch.amp.GradScaler("cuda")`. DeBERTa's disentangled attention is numerically unstable under bf16 — gradients diverge to NaN after the first optimizer step, and every subsequent forward pass also returns NaN. fp16 with GradScaler prevents this.

**Differential LR**: backbone uses `peak_lr × 0.1 = 2e-6`; the randomly initialised `WafClassifier` head uses `peak_lr = 2e-5`. This prevents the pretrained backbone from being disrupted by the large initial head gradients.

**Epoch count**: computed from `max_steps // len(train_loader)` so the same config works regardless of dataset size.

**Early stopping**: on `macro_f1`, patience from `training.track_a.early_stopping_patience`.

---

### `02_track_a_small.py` — Track A small fine-tuning

**Inputs:** `data/splits/train.parquet`, `data/splits/val.parquet`  
**Outputs:** `models/track_a/small/latest`

Structurally identical to script 01 but uses `cfg.model.track_a_small.base_model` (default: `google/bert_uncased_L-2_H-128_A-2`, Google's official BERT-tiny: 2 layers, hidden=128, 2 heads). Key difference: `backbone_lr_multiplier=0.3` (more permissive than 0.1 for 01, appropriate for a smaller backbone that needs faster adaptation). Uses bf16 autocast without GradScaler — BERT weights are stored in fp32, so there is no fp16 gradient issue.

`google/bert_uncased_L-2_H-128_A-2` is preferred over `prajjwal1/bert-tiny` (same architecture) because the latter's HuggingFace repository has broken tokenizer files that fail to instantiate regardless of `use_fast` setting.

The script owns its complete training loop independently, ensuring MLflow run names correctly identify small-model runs in the experiment UI.

---

### `03_track_b_99m.py` — Custom 99M Transformer, supervised training

**Inputs:** `tokenizers/track_b/tokenizer.json`, `data/splits/train.parquet`, `data/splits/val.parquet`  
**Outputs:** `models/track_b/99m/latest`

Trains a `TinyTransformerEncoder` from scratch using the Track B BPE vocabulary. No pretrained weights are loaded.

**`TinyTransformerEncoder`** — lightweight BERT-style encoder:
- Embedding stack: word embeddings + absolute position embeddings + token-type embeddings, combined and normalised before the encoder.
- `nn.TransformerEncoder` with `norm_first=True` (Pre-LayerNorm) for training stability from random initialisation. Pre-LN avoids vanishing gradients in deep transformers by normalising before each sub-layer rather than after.
- Padding mask convention: `src_key_padding_mask` expects `True` where tokens should be ignored. The mask is derived from `attention_mask == 0`.
- Weight initialisation: truncated normal (σ=0.02) for Linear and Embedding layers, ones/zeros for LayerNorm, zeros for padding embeddings.

**`TrackB99MModel`** — wraps `TinyTransformerEncoder` and `WafClassifier`. Exposes `get_hidden_states()` which returns the `[CLS]` token embedding (position 0) without the classification head — used by `03b_distill_track_b.py` for hidden-state alignment.

**Tokenizer coupling**: `vocab_size` is read directly from the loaded `HttpTokenizer`, not from config, to ensure the embedding table is always sized to the actual trained vocabulary.

**No backbone LR multiplier**: `backbone_lr_multiplier=1.0` — there is no pretrained backbone, all weights start random and should learn at the full LR.

Config used: `training.teacher` (same as `WafEncoder` training in Stage 6 — peak_lr=1e-4, accum_steps=4, max_steps=50 000, bf16).

---

### `03b_distill_track_b.py` — Cross-tokenizer knowledge distillation

**Inputs:** `models/track_a/large/latest` (frozen teacher), `tokenizers/track_b/tokenizer.json`, `models/track_b/99m/latest` (student warm-start, optional), `data/splits/train.parquet`, `data/splits/val.parquet`  
**Outputs:** `models/track_b/99m/latest` (overwrites supervised checkpoint with distilled version)

Transfers knowledge from the fine-tuned DeBERTa-v3 teacher into the Track B 99M student. The key complication: teacher and student use **different tokenizers** (DeBERTa WordPiece vs Track B BPE). Standard distillation assumes a shared tokenizer; this script handles the dual-tokenization case.

**`DistillDataset`** — each sample is tokenized twice: once with `HttpTokenizer` (student input) and once with the HuggingFace DeBERTa tokenizer (teacher input). The teacher and student never see the same token IDs, but they do produce the same-shaped logits `(batch, num_labels)`.

**`distill_collate_fn`** — pads student and teacher sequences independently to their own batch-maximum lengths, producing four tensors per batch: `student_input_ids`, `student_attention_mask`, `teacher_input_ids`, `teacher_attention_mask`.

**`FrozenDeBERTaTeacher`** — loads the Stage 5.1 checkpoint and freezes all parameters. All forward passes run under `torch.no_grad()`. The teacher is never updated.

**Distillation loss**:
```
L = α · CE(student_logits, hard_labels)
  + (1 − α) · T² · KL( softmax(student / T) ‖ softmax(teacher / T) )
```
The T² factor keeps gradient magnitudes consistent as temperature changes — without it, increasing T would shrink gradients quadratically. `alpha` and `temperature` come from `training.distillation` config (`alpha_hard=0.3`, `temperature=4.0`).

**Student warm-start**: if a supervised checkpoint exists at `models/track_b/99m/latest` (from script 03), it is loaded before distillation begins. This gives the student a head start and typically improves final quality versus distilling from random initialisation. If the checkpoint is absent, a warning is logged and distillation proceeds from scratch.

**Validation**: a separate student-only DataLoader (BPE tokens, no teacher IDs) is constructed for the standard `evaluate()` call. The dual-tokenized `val_loader` cannot be passed directly to `evaluate()` because its batch keys don't match the student model's `forward()` signature.

**MLflow run name**: `track_b_99m_distilled` — distinct from the `track_b_99m` run produced by script 03, so both appear separately in the experiment UI.

Note: `hidden_mse_weight` is set to `0.0` in config (distillation.hidden_mse_weight). Hidden-state alignment would require teacher and student to share hidden dimensionality, which is not the case here (DeBERTa-v3 has d_model=768; so does the 99M student, but the representation spaces are incompatible across tokenizers).

---

## Data flow

```
data/splits/train.parquet
data/splits/val.parquet
        │
        ├──▶ 01_track_a_large.py ──▶ models/track_a/large/latest  (337M)
        │       (DeBERTa-v3-base, fp16+GradScaler, LR differential)
        │              │
        │              │ frozen teacher
        │              ▼
        ├──▶ 02_track_a_small.py ──▶ models/track_a/small/latest  (~4M)
        │       (BERT-tiny, bf16)
        │
        └──▶ 03_track_b_99m.py  ──▶ models/track_b/99m/latest  (99M, supervised)
                │       (TinyTransformer from scratch, BPE vocab)
                │
                │  warm-start          frozen DeBERTa teacher (from 01)
                ▼                               │
        03b_distill_track_b.py ◀───────────────┘
                │
                │  quality transfer (337M → 99M, cross-tokenizer)
                ▼
        models/track_b/99m/latest  (99M, distilled)  [overwrites supervised latest]
                │
                │  ← Stage 5 ends here
                │
                ▼
        STAGE 6: compression distillation (99M → 10M, same tokenizer)
                │
                ▼
        models/student/  (10M, deployable, p99 <5ms)
```

`03b` is optional — if skipped, `models/track_b/99m/latest` points to the supervised checkpoint and Stage 6 distils from that instead. The individual `step_NNNNNN/` checkpoint directories from script 03 are preserved on disk; only the `latest` symlink is updated by 03b.

---

## Configuration

All Stage 5 settings live under `model` and `training` in `config/pipeline.yaml`.

**Model configs** (`model.*`):

```yaml
model:
  track_a_large:
    base_model:  "microsoft/deberta-v3-base"
    num_labels:  2
    dropout:     0.1
    output_dir:  "models/track_a/large"

  track_a_small:
    base_model:  "google/bert_uncased_L-2_H-128_A-2"
    num_labels:  2
    dropout:     0.1
    output_dir:  "models/track_a/small"

  track_b_99m:
    d_model:     768
    n_layers:    13
    n_heads:     12
    d_ff:        3072      # 4 × d_model
    dropout:     0.1
    num_labels:  2
    output_dir:  "models/track_b/99m"
```

**Training configs** (`training.*`):

```yaml
training:
  track_a:                           # used by scripts 01 and 02
    batch_size:            8
    grad_accum_steps:      8         # effective batch = 64
    max_steps:             20000
    warmup_ratio:          0.06      # warmup_steps = max_steps × ratio = 1200
    peak_lr:               2.0e-5   # backbone gets ×0.1 (01) or ×0.3 (02)
    weight_decay:          0.01
    grad_clip_norm:        1.0
    early_stopping_patience: 5

  teacher:                           # used by script 03
    batch_size:            8
    grad_accum_steps:      4         # effective batch = 32
    max_steps:             50000
    warmup_ratio:          0.06
    peak_lr:               1.0e-4
    weight_decay:          0.01
    grad_clip_norm:        1.0
    early_stopping_patience: 10

  distillation:                      # used by script 03b
    batch_size:            128
    grad_accum_steps:      8
    max_steps:             40000
    warmup_ratio:          0.06
    peak_lr:               2.0e-4
    alpha_hard:            0.3       # CE weight (α)
    alpha_soft:            0.7       # KL weight (1 − α)
    temperature:           4.0       # T
    hidden_mse_weight:     0.0       # disabled (cross-tokenizer)
```

---

## Running the stage

```bash
# Primary path — custom tokenizer + Track B 99M
make train_b_99m

# Individual scripts
python stages/5_teacher_training/01_track_a_large.py --config config/pipeline.yaml
python stages/5_teacher_training/02_track_a_small.py --config config/pipeline.yaml
python stages/5_teacher_training/03_track_b_99m.py   --config config/pipeline.yaml
python stages/5_teacher_training/03b_distill_track_b.py --config config/pipeline.yaml

# Force re-run even if checkpoint already exists
python stages/5_teacher_training/03_track_b_99m.py --force
```

All scripts are idempotent by default: if `checkpoint_meta.json` already exists at the configured `output_dir`, the script exits immediately with a log message. Pass `--force` to override.

Scripts must be run from the repo root so that `from checkpoint_utils import ...` and `from train_utils import ...` resolve correctly (Stage 5 modules are not installed as a package).

---

## What to check after running

| Check | Where |
|---|---|
| Training loss curve | MLflow UI (`make ui`) → experiment runs `track_a_large`, `track_b_99m`, `track_b_99m_distilled` |
| Best macro-F1 per model | `models/<track>/latest/checkpoint_meta.json` → `macro_f1` |
| Non-finite loss warnings | Console output; `[warn] non-finite loss at batch N` — isolated warnings are normal, 20 consecutive means divergence |
| Symlink correctness | `ls -la models/track_a/large/` and `models/track_b/99m/` — `latest` should point to the highest-F1 step directory |
| Teacher checkpoint for 03b | `models/track_a/large/latest` must exist before running `03b_distill_track_b.py` |

Early stopping fires when `macro_f1` has not improved for `patience` epochs. If training ends suspiciously early (epoch 1–2), check the val split for class imbalance and confirm `data/splits/val.parquet` was produced by Stage 3's stratified split.
