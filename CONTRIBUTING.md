# Contributing

Thanks for working on ai-waf-v2. This guide covers the development workflow and
conventions. For running the pipeline itself, see
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Setup

```bash
touch .env                 # required by every task (may be empty)
python run.py setup        # creates .venv and installs editable + dev deps
```

`python run.py <task>` works everywhere (Windows/macOS/Linux); `make <task>` is the
equivalent if you have `make`. `python run.py --list` shows all tasks.

## Before you open a PR

```bash
python run.py quality      # ruff check --fix + ruff format + mypy on stages/
python run.py test         # full suite — synthetic data, no GPU/network needed
```

Both must pass. Tests live entirely in `tests/test_all.py` and use small in-memory
synthetic data, so they run fast on CPU. Scope a run while iterating:

```bash
.venv/bin/python -m pytest tests/test_all.py -k Metrics -v
```

## Conventions

- **No magic numbers.** All hyperparameters and paths live in
  `config/pipeline.yaml`, loaded through the Pydantic models in
  `ai_waf_v2/utils/config.py`. Add a config field rather than hardcoding.
- **Library vs stages.** Reusable logic belongs in `ai_waf_v2/` (and gets a test);
  `stages/` scripts should only orchestrate, do I/O, and log.
- **Stage scripts are idempotent.** Use `require_inputs` / `check_output` from
  `ai_waf_v2/utils/pipeline.py` so a script skips when its outputs exist and
  re-runs under `--force`.
- **Digit-prefixed stage dirs aren't importable.** If two scripts in a stage need
  to share a class, put it in a non-numeric module (e.g. `track_b_model.py`) and
  import that.
- **Data flows as `HttpRecord` / Parquet.** Don't invent ad-hoc record formats;
  extend the schema in `ai_waf_v2/data/schema.py` if a field is genuinely missing.
- **Log through the helpers.** `configure_root()` once per entry point, then
  `get_logger(__name__)`; log metrics to MLflow via `ai_waf_v2/utils/mlflow_utils.py`.
- **Keep the runners in sync.** Any new stage script or target must be added to
  **both** the `Makefile` and `run.py` (they expose an identical task set).

## Adding a pipeline stage script

1. Add the script under the appropriate `stages/N_*/` directory, numbered in run
   order, with a `--config` (and usually `--force`) argument.
2. Read inputs and write outputs through the `HttpRecord`/Parquet schema.
3. Wire it into the `Makefile` and `run.py` as a task (and into any composite
   target it belongs to).
4. Document it in [docs/stages/](docs/stages/) and, if it changes the public API,
   in [docs/API.md](docs/API.md).
5. Add or update tests in `tests/test_all.py` for any new library code.

## Commits & changelog

- Keep commits focused; describe the *why* in the body.
- Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for notable changes.
- Never commit secrets — `.env` is local-only and git-ignored.
