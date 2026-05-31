# HuggingFace Hub Integration

Artifacts produced by the pipeline can be pushed to a private HuggingFace organisation (or personal account) for safe-keeping and sharing with collaborators.

Push happens in two ways:

- **Per-stage** — each stage script accepts `--save` and pushes its own output immediately after completing. This gives incremental backups as the pipeline runs.
- **All-at-once** — `make push_hub` pushes every artifact that currently exists on disk in one go.

---

## Setup

### 1. Create a HuggingFace account and organisation

Sign up at [huggingface.co](https://huggingface.co). Create an organisation at **Settings → Organisations → New organisation** if you want a shared namespace (e.g. `my-lab`).

### 2. Generate an access token

Go to **Settings → Access Tokens → New token**. Select **Write** scope so the token can create repos and push files.

### 3. Authenticate

```bash
# Option A — environment variable (recommended for scripts and CI)
export HF_TOKEN=hf_...

# Option B — interactive login (stores credentials in ~/.cache/huggingface)
huggingface-cli login
```

### 4. Set your organisation slug

```bash
export HF_ORG=my-lab          # or your personal username
```

Or set it permanently in `config/pipeline.yaml`:

```yaml
huggingface:
  org: "my-lab"
```

### 5. Install the client library

```bash
pip install huggingface_hub
```

---

## Repository layout

Each artifact gets its own private repo. Default names are set in `config/pipeline.yaml` under `huggingface.repos` and can be overridden freely:

| Config key | Default repo name | Contains |
|---|---|---|
| `tokenizer` | `waf-tokenizer-track-b` | Custom HTTP-aware BPE tokenizer |
| `teacher` | `waf-teacher-99m` | 99M teacher checkpoint |
| `student` | `waf-student-10m` | 10M student checkpoint + ONNX export |
| `dataset_synthesis` | `waf-dataset-synthesis` | Synthesized attack payloads (Stage 3.1) |
| `dataset_framed` | `waf-dataset-framed` | Framed records — attack + benign (Stage 3.3) |
| `dataset_base` | `waf-dataset-base` | Pre-augmentation splits |
| `dataset_aug` | `waf-dataset-augmented` | Final augmented splits |

All repos are created as **private** by default (`huggingface.private: true`). Set to `false` to make them public.

---

## Per-stage push

Add `--save` to any supported stage script. The script runs normally, then pushes its output folder to the Hub.

```bash
# Stage 3.1 — push synthesized attacks after generation completes
python stages/3_data_augmentation/01_attack_synthesis.py --save

# Stage 3.3 — push framed records after framing completes
python stages/3_data_augmentation/03_request_framing.py --save
```

### `--revision` — tagging versions

Use `--revision` to push to a named branch or tag. This is the cleanest way to keep a pre-augmentation and a post-augmentation version in the same repo:

```bash
# Push pre-aug baseline model
python stages/3_data_augmentation/01_attack_synthesis.py --save --revision v1.0-base

# Re-run after augmentation; push to a different revision
python stages/3_data_augmentation/01_attack_synthesis.py --force --save --revision v1.0-aug
```

Collaborators can then pull a specific version:

```python
from huggingface_hub import snapshot_download
snapshot_download("my-lab/waf-dataset-synthesis", revision="v1.0-aug", repo_type="dataset")
```

### `--dry-run` — preview without uploading

```bash
python stages/3_data_augmentation/01_attack_synthesis.py --save --dry-run
```

Logs what would be uploaded (file count, total MB, target repo and revision) without making any API calls.

### Via `make`

Pass `HF_SAVE=1` to enable push for any augmentation target. Combine with `HF_REVISION` to tag the revision:

```bash
make data_augment_synthesis HF_SAVE=1
make data_augment_framing   HF_SAVE=1 HF_REVISION=v1.0-aug
make data_augment_all       HF_SAVE=1 HF_REVISION=v1.0-aug
```

---

## Push everything at once

Use `make push_hub` when you want to sync all artifacts that currently exist on disk in one shot:

```bash
# Push all models + all datasets
make push_hub

# Push only datasets
make push_hub HF_ONLY=datasets

# Push only models (tokenizer, teacher, student)
make push_hub HF_ONLY=models

# Push to a specific revision
make push_hub HF_REVISION=v1.0-aug

# Dry-run preview
make push_hub_dry
```

Or run the script directly:

```bash
python stages/7_evaluation/17_push_to_hub.py --only datasets --revision v1.0-aug
python stages/7_evaluation/17_push_to_hub.py --dry-run
```

---

## Pulling artifacts (collaborators)

```python
from huggingface_hub import snapshot_download

# Download a dataset split
path = snapshot_download(
    repo_id="my-lab/waf-dataset-augmented",
    repo_type="dataset",
    revision="main",        # or "v1.0-aug"
)

# Download the student model
path = snapshot_download(
    repo_id="my-lab/waf-student-10m",
    repo_type="model",
)
```

Or from the shell:

```bash
huggingface-cli download my-lab/waf-dataset-augmented --repo-type dataset --local-dir data/splits/
huggingface-cli download my-lab/waf-student-10m --repo-type model --local-dir models/student/
```

---

## Configuration reference

All HuggingFace settings live in `config/pipeline.yaml` under `huggingface:`. Every value supports `${ENV_VAR}` expansion.

```yaml
huggingface:
  org:     "${HF_ORG}"   # required — your org/username slug
  private: true          # set false to make repos public
  repos:
    tokenizer:         "waf-tokenizer-track-b"
    teacher:           "waf-teacher-99m"
    student:           "waf-student-10m"
    dataset_base:      "waf-dataset-base"
    dataset_aug:       "waf-dataset-augmented"
    dataset_synthesis: "waf-dataset-synthesis"
    dataset_framed:    "waf-dataset-framed"
```

To rename a repo, change the value here — the next push will create (or update) a repo with the new name. Old repos are not deleted automatically.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HuggingFace org not set` warning, push skipped | Export `HF_ORG=your-org` |
| `huggingface_hub not installed` | `pip install huggingface_hub` |
| `401 Unauthorized` | Token missing or expired — re-run `huggingface-cli login` or re-export `HF_TOKEN` |
| `403 Forbidden` on a private repo | Token needs **Write** scope |
| Push silently skipped | Source folder doesn't exist yet — run the upstream stage first |
