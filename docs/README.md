# ai-waf-v2 — Documentation

Central index for all project documentation. Every doc has a **📖 Docs** menu at the
top linking back here.

## Start here

| Doc | What it covers |
|---|---|
| [Repo README](../README.md) | Project overview, why, and quick start |
| [User Guide](USER_GUIDE.md) | Fresh clone → trained, evaluated model; every command, env knob, report, and the full `run.py` task reference |
| [Architecture](ARCHITECTURE.md) | Pipeline DAG, core library design, data flow, config system |
| [API Reference](API.md) | The `ai_waf_v2` library public API |
| [Model Card](MODEL_CARD.md) | Intended use, training data, metrics, limitations |

## Pipeline stages

| Stage | Deep dive |
|---|---|
| **Overview** | [Pipeline Stages Reference](stages/stages.md) |
| 1 | [Data Acquisition & Curation](stages/stage1_data_acquisition.md) |
| 2 | [Baselines](stages/stage2_baselines.md) |
| 3 | [Data Augmentation](stages/stage3_data_augmentation.md) |
| 4 | [Tokenization](stages/stage4_tokenization.md) |
| 5 | [Teacher Training](stages/stage5_teacher_training.md) |
| 6 | [Distillation & Compression](stages/stage6_distillation_and_compression.md) |
| 7 | [Evaluation](stages/stage7_evaluation.md) |

## Attack taxonomy & synthesis

| Doc | Topic |
|---|---|
| [Attack Synthesis](attacks/attack_synthesis.md) | How synthetic attack payloads are generated |
| [XSS Attacks](attacks/xss-attack.md) | Cross-site scripting taxonomy |
| [CMDi Taxonomy](attacks/cmdi-attack-taxonomy.md) | Command-injection taxonomy |

## Project & process

| Doc | Topic |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Dev workflow, quality checks, conventions |
| [Changelog](../CHANGELOG.md) | Notable changes |
| [CLAUDE.md](../CLAUDE.md) | Codebase guide for the Claude Code agent |
