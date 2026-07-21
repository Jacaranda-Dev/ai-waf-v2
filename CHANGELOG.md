# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Report path registry** (`ai_waf_v2/utils/reports.py`): reports are now organised
  by stage — `reports/<N>_<stage>/<metrics|latency|figures>/<NN>_<name>` — instead of
  a flat `reports/metrics/`. Every stage script resolves paths through
  `report_path(...)`/`report_dir(...)`, so the layout lives in one `REPORTS` dict and
  cross-stage readers/writers can't drift.

- `ai_waf_v2/augment/` — recursive **PCFG** attack-payload engine (`pcfg.py`) and
  per-class **payload validity** checks (`validity.py`). Stage 3.1 grammars now
  live as data in `config/pcfg_grammars.yaml` (path `augmentation.pcfg_grammars_path`);
  `PcfgGenerator` samples them where present and falls back to the flat
  `GrammarGenerator` templates otherwise. Covered by new tests in `tests/test_all.py`
  (grammar termination, YAML loader, sampler fallback/top-up, validity).
- Stage 3.4 quality gate gained a per-class **structural validity** pass
  (`ClassValidity`, config `augmentation.filtering.class_validity_check`) that
  rejects payloads no longer matching their `attack_class` — e.g. malformed
  recursive-grammar output. The gate now runs five passes instead of four.
- Richer recursive **`xss`** grammar in `config/pcfg_grammars.yaml` (HTML-text,
  attribute-breakout and event-handler-injection contexts; `javascript:` URIs;
  nested tags; chained JS). Docs: an "Authoring PCFG attack grammars" guide in
  `docs/stages/stage3_data_augmentation.md`.
- Recursive **`cmdi`** grammar in `config/pcfg_grammars.yaml` (separator/newline
  chaining, `$(…)`/backtick command substitution, reverse shells, redirections).
- Recursive **`ssti`** grammar (Jinja/`${}`/`#{}`/ERB/Spring engines; recursive
  attribute/index gadget chains) and a shared **`lfi`** / **`path_traversal`**
  grammar (variable-depth `../` traversal, `php://`/`data://` wrappers, null-byte
  suffixes) defined once via a YAML anchor.
- **Token↔label leakage measurement** (`ai_waf_v2/eval/leakage.py`, Stage 3.8
  `make data_leakage`): `token_leakage` ranks tokens by mutual information /
  `P(malicious|token)` and flags frequent, near-deterministic *incidental*
  (host/number/id-shaped) predictors — the shortcuts a WAF must not learn.
  `counterfactual_auc_delta` / `swap_fillers` re-randomise incidental values and
  re-score (small AUC delta ⇒ relies on structure; large ⇒ memorised constants).
  The stage also runs a model-free counterfactual (swap fillers, recompute; MI
  drops if leakage is filler-driven) and writes `reports/metrics/token_leakage.json`.
- **Label-neutral fillers** (`ai_waf_v2/augment/fillers.py`): grammars now emit
  `§NAME§` placeholders for incidental values (hosts, ports, ids, numbers, JS
  bodies) instead of hardcoded literals like `evil.com`/`4444`, filled at synthesis
  from high-cardinality generators with optional `§NAME#TAG§` coreference. The same
  generators back benign URL/path params in Stage 3.3 (replacing fixed
  `example.com` hosts), so incidental tokens carry no label signal and the model
  learns attack structure rather than memorisable constants. `f:` is now reserved
  for small structural vocab (`xss_evt`, `xss_tag`); the LLM textbook-host scrub
  now shares the same distribution.
- `run.py` — a cross-platform, dependency-free task runner mirroring every
  `Makefile` target, so the pipeline can be driven without `make` (Windows/macOS).
- Documentation set: top-level `README`, `docs/USER_GUIDE.md`,
  `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/MODEL_CARD.md`, `CONTRIBUTING.md`,
  and this changelog.
- `Makefile` / `run.py` targets for previously unwired scripts: `train_teacher`
  (Stage 6.0), `distill_track_b` (Stage 5.3b), `distill_canary` (Stage 6.4), and
  `deploy_reco` (Stage 7.16).

### Changed
- Consolidated attack documentation into `docs/attacks/`; removed superseded
  top-level docs (kept `docs/stages/` and `docs/attacks/`).
- Distillation targets renamed/retargeted to match real scripts: `distill_quant`
  → `distill_calibrate`; `distill_export` now calls `05_export_and_bench.py`.

### Fixed
- `Makefile` referenced nonexistent scripts (Stage 5 threshold-calibration/canary,
  Stage 6 quant/export) and passed `--mlflow-run-name` to teacher-training scripts
  that don't accept it. All targets now point at scripts that exist.
- `test_core` / `test_data` / `test_distill` pointed at test files that don't
  exist; they now scope `tests/test_all.py` with `-k` filters.

---

<!--
Template for a tagged release:

## [x.y.z] - YYYY-MM-DD
### Added / Changed / Deprecated / Removed / Fixed / Security
-->
