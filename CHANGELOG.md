# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
