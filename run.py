#!/usr/bin/env python3
"""
run.py — cross-platform task runner for ai-waf-v2.

A dependency-free alternative to the Makefile for users who can't (or don't want
to) run `make` — notably on Windows and macOS. Pure standard library; no install.

    python run.py --list                # show every task
    python run.py train_b_99m           # run one task (and its dependencies)
    python run.py baselines tokenize_b  # run several, in order
    python run.py -n eval_only          # dry-run: print commands, run nothing

Tasks mirror the Makefile targets one-for-one. Like make, each task runs its
dependencies first, and a dependency shared by several tasks runs only once per
invocation.

Configuration knobs (env vars, optionally supplied via a local `.env` file which
this script loads automatically — so unlike the Makefile, a missing `.env` is
*not* a hard error):

    MODE=full|research      setup: dependency extras to install (default: full)
    HF_REVISION, HF_ONLY, HF_SAVE     HuggingFace push / save options
    LOCAL_LLM_PATH, LLM_PROVIDER      augmentation LLM options
    PCAP_DIR                          traffic profiler input
    ANTHROPIC_API_KEY, GOOGLE_API_KEY, HF_TOKEN, MLFLOW_TRACKING_URI, ...
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = "config/pipeline.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────
def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from .env into os.environ (real env wins)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def venv_python() -> str:
    """Path to the project venv interpreter, or the current one as fallback."""
    if platform.system() == "Windows":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


PYTHON = venv_python()


# ──────────────────────────────────────────────────────────────────────────────
# Command helpers — these build argv lists; nothing is run until run_cmd()
# ──────────────────────────────────────────────────────────────────────────────
def stage(script: str, *extra: str) -> list[str]:
    """A stage script invocation: <python> <script> --config <cfg> [extra...]."""
    return [PYTHON, script, "--config", CFG, *extra]


def opt(flag: str, value: str | None) -> list[str]:
    """[flag, value] when value is truthy, else []  (mirrors $(if ...) in make)."""
    return [flag, value] if value else []


def flag_if(flag: str, present: bool) -> list[str]:
    return [flag] if present else []


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def has(name: str) -> bool:
    return bool(os.environ.get(name))


def stamp(prefix: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M}"


# ──────────────────────────────────────────────────────────────────────────────
# Side-effecting steps (file ops the make recipes do with cp/mkdir/rm)
# ──────────────────────────────────────────────────────────────────────────────
def _snapshot_pre_aug_splits(dry: bool) -> None:
    src, dst = ROOT / "data" / "splits", ROOT / "data" / "splits_pre_aug"
    print(f"# copy {src}/*.parquet -> {dst}/")
    if dry:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for parquet in src.glob("*.parquet"):
        shutil.copy2(parquet, dst / parquet.name)


def _rmtree_globs(*globs: str):
    def _do(dry: bool) -> None:
        for pattern in globs:
            for match in ROOT.glob(pattern):
                print(f"# rm -rf {match}")
                if dry:
                    continue
                if match.is_dir():
                    shutil.rmtree(match, ignore_errors=True)
                else:
                    match.unlink(missing_ok=True)

    return _do


def _stop_ui(dry: bool) -> None:
    if dry:
        print("# stop mlflow ui")
        return
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/F", "/IM", "mlflow.exe"], check=False)
    else:
        rc = subprocess.run(["pkill", "-f", "mlflow ui"], check=False).returncode
        print("MLflow UI stopped." if rc == 0 else "No MLflow UI process found.")


def _do_setup(dry: bool) -> None:
    mode = env("MODE", "full")
    extras = ".[research]" if mode == "research" else ".[research,full]"
    desc = "Python (ai-waf-v2 Research)" if mode == "research" else "Python (ai-waf-v2 Full Stack)"
    print(f"=== Initializing environment in [{mode}] mode ===")
    run_cmd([sys.executable, "-m", "venv", ".venv"], dry)
    vpy = venv_python() if not dry else PYTHON
    # after venv creation the interpreter exists; recompute explicitly
    if not dry:
        vpy = str(
            (ROOT / ".venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))
        )
    run_cmd([vpy, "-m", "pip", "install", "--upgrade", "pip"], dry)
    run_cmd([vpy, "-m", "pip", "install", "-e", extras], dry)
    run_cmd(
        [vpy, "-m", "ipykernel", "install", "--user", "--name=ai-waf-v2", f"--display-name={desc}"],
        dry,
    )
    print("Setup complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Task table.  Each task: {"desc", "deps": [names], "steps": [step, ...]}
# A step is either a list[str] command, or a callable(dry: bool) -> None.
# Commands that depend on env vars / timestamps are wrapped in lambdas so they
# resolve at run time, not import time.
# ──────────────────────────────────────────────────────────────────────────────
S1 = "stages/1_data_acquisition_and_curation"
S2 = "stages/2_baselines"
S3 = "stages/3_data_augmentation"
S4 = "stages/4_tokenization"
S5 = "stages/5_teacher_training"
S6 = "stages/6_distillation_and_compression"
S7 = "stages/7_evaluation"

TASKS: dict[str, dict] = {
    # ── environment / quality / tests ──
    "setup": {"desc": "Create .venv and install deps (MODE=full|research)", "steps": [_do_setup]},
    "lint": {
        "desc": "Ruff check --fix + format",
        "steps": [[PYTHON, "-m", "ruff", "check", ".", "--fix"], [PYTHON, "-m", "ruff", "format", "."]],
    },
    "typecheck": {"desc": "Mypy on stages/", "steps": [[PYTHON, "-m", "mypy", "stages/"]]},
    "quality": {"desc": "lint + typecheck", "deps": ["lint", "typecheck"]},
    "test": {"desc": "Full test suite", "steps": [[PYTHON, "-m", "pytest", "tests/test_all.py"]]},
    "test_core": {"desc": "Model architecture tests (encoder + classifier)",
                  "steps": [[PYTHON, "-m", "pytest", "tests/test_all.py", "-k", "WafEncoder or WafClassifier"]]},
    "test_data": {"desc": "Data schema + collator tests",
                  "steps": [[PYTHON, "-m", "pytest", "tests/test_all.py", "-k", "HttpRecord or WafCollator"]]},
    "test_distill": {"desc": "Distillation loss + metrics tests",
                     "steps": [[PYTHON, "-m", "pytest", "tests/test_all.py", "-k", "DistillationLoss or Metrics"]]},

    # ── stage 1 ──
    "data_collect": {
        "desc": "Stage 1.1-1.2: acquire, normalize, dedup",
        "steps": [
            stage(f"{S1}/01_acquire_and_normalize.py"),
            stage(f"{S1}/02_cross_dataset_dedup.py"),
        ],
    },
    "data_analyze": {
        "desc": "Stage 1.3-1.4: corpus coverage report",
        "deps": ["data_collect"],
        "steps": [stage(f"{S1}/03_generate_corpus_report.py")],
    },

    # ── stage 2 ──
    "baselines": {
        "desc": "Stage 2: pre-augmentation baselines (split snapshot + CRS + classical)",
        "steps": [
            stage(f"{S2}/00_stratified_split.py"),
            _snapshot_pre_aug_splits,
            stage(f"{S2}/01_define_slos.py"),
            stage(f"{S2}/02_modsecurity_crs.py"),
            stage(f"{S2}/03_classical_ml_baseline.py", "--split-dir", "data/splits_pre_aug",
                  "--out-file", "baselines.json", "--phase", "pre_aug"),
        ],
    },
    "baselines_post_aug": {
        "desc": "Stage 7: re-run classical baselines on augmented splits",
        "steps": [stage(f"{S2}/03_classical_ml_baseline.py", "--out-file",
                        "baselines_post_aug.json", "--phase", "post_aug")],
    },

    # ── stage 3 ──
    "data_augment_synthesis": {
        "desc": "Stage 3.1: attack synthesis (grammar/mutator/tamper/local LLM)",
        "steps": [lambda dry: run_cmd(
            stage(f"{S3}/01_attack_synthesis.py",
                  *opt("--model-path", env("LOCAL_LLM_PATH")),
                  *flag_if("--save", has("HF_SAVE")),
                  *opt("--revision", env("HF_REVISION"))), dry)],
    },
    "data_augment_benign": {
        "desc": "Stage 3.2: traffic distribution profiler (PCAP alignment)",
        "steps": [lambda dry: run_cmd(
            stage(f"{S3}/02_traffic_profiler.py", *opt("--pcap-dir", env("PCAP_DIR"))), dry)],
    },
    "data_augment_framing": {
        "desc": "Stage 3.3: request framing (reframe + benign REST + optional cloud LLM)",
        "steps": [lambda dry: run_cmd(
            stage(f"{S3}/03_request_framing.py",
                  *opt("--provider", env("LLM_PROVIDER")),
                  *flag_if("--save", has("HF_SAVE")),
                  *opt("--revision", env("HF_REVISION"))), dry)],
    },
    "data_filter": {
        "desc": "Stage 3.4: quality gate (format/UNK/dedup/label consistency)",
        "steps": [stage(f"{S3}/04_quality_gate.py")],
    },
    "data_validate": {
        "desc": "Stage 3.5-3.6: augmentation probe + taxonomy inventory",
        "steps": [stage(f"{S3}/05_augmentation_probe.py"), stage(f"{S3}/06_taxonomy_inventory.py")],
    },
    "data_split": {
        "desc": "Stage 3.7: stratified split of augmented corpus",
        "steps": [stage(f"{S3}/07_stratified_split.py")],
    },
    "data_augment_all": {
        "desc": "Stage 3: full augmentation pipeline",
        "deps": ["data_augment_synthesis", "data_augment_benign", "data_augment_framing",
                 "data_filter", "data_validate", "data_split"],
    },

    # ── stage 4 ──
    "tokenize_a": {
        "desc": "Stage 4A: augment pretrained vocab + OOV",
        "steps": [stage(f"{S4}/01_augment_pretrained_vocab.py"), stage(f"{S4}/02_measure_oov_track_a.py")],
    },
    "tokenize_b": {
        "desc": "Stage 4B: custom BPE from scratch + OOV",
        "steps": [stage(f"{S4}/03_train_custom_bpe.py"), stage(f"{S4}/04_measure_oov_track_b.py")],
    },
    "tokenize_eval": {
        "desc": "Stage 4: tokenizer comparison report",
        "steps": [stage(f"{S4}/05_compare_tokenizers.py")],
    },
    "tokenize": {"desc": "Stage 4: both tracks + comparison",
                 "deps": ["tokenize_a", "tokenize_b", "tokenize_eval"]},

    # ── stage 5 ──
    "train_a_large": {
        "desc": "Stage 5.1: Track A DeBERTa-v3-base fine-tune",
        "steps": [stage(f"{S5}/01_track_a_large.py")],
    },
    "train_a_small": {
        "desc": "Stage 5.2: Track A small pretrained fine-tune",
        "steps": [stage(f"{S5}/02_track_a_small.py")],
    },
    "train_b_99m": {
        "desc": "Stage 5.3: Track B 99M encoder from scratch",
        "steps": [stage(f"{S5}/03_track_b_99m.py")],
    },
    "distill_track_b": {
        "desc": "Stage 5.3b: cross-track distillation (Track A teacher -> Track B student)",
        "steps": [stage(f"{S5}/03b_distill_track_b.py")],
    },

    # ── stage 6 ──
    "train_teacher": {
        "desc": "Stage 6.0: train/resume 99M teacher (re-entry point)",
        "steps": [stage(f"{S6}/00_train_teacher_99m.py")],
    },
    "distill_train": {
        "desc": "Stage 6.1-6.2: knowledge distillation -> student",
        "steps": [
            stage(f"{S6}/01_student_arch.py"),
            lambda dry: run_cmd(
                stage(f"{S6}/02_distill_train.py", "--mlflow-run-name", stamp("student_distill")), dry),
        ],
    },
    "distill_calibrate": {
        "desc": "Stage 6.3: post-training temperature calibration",
        "steps": [stage(f"{S6}/03_student_calibrate.py")],
    },
    "distill_canary": {
        "desc": "Stage 6.4: student canary / memorisation check",
        "steps": [stage(f"{S6}/04_student_canary.py")],
    },
    "distill_export": {
        "desc": "Stage 6.5: ONNX + TorchScript export & latency bench",
        "steps": [stage(f"{S6}/05_export_and_bench.py")],
    },
    "distill": {"desc": "Stage 6: distill + calibrate + canary + export",
                "deps": ["distill_train", "distill_calibrate", "distill_canary", "distill_export"]},

    # ── stage 7 ──
    "eval_detection": {"desc": "Stage 7.1: detection efficacy",
                       "steps": [stage(f"{S7}/01_detection_metrics.py")]},
    "eval_latency": {
        "desc": "Stage 7.2-7.3: latency/throughput + memory footprint",
        "steps": [
            stage(f"{S7}/02_latency_bench.py", "--batch-sizes", "1", "8", "32", "64", "256",
                  "--devices", "gpu", "cpu"),
            stage(f"{S7}/03_memory_footprint.py"),
        ],
    },
    "eval_adversarial": {
        "desc": "Stage 7.4-7.6: adversarial robustness",
        "steps": [stage(f"{S7}/04_evasion_payloads.py"), stage(f"{S7}/05_obfuscation_robustness.py"),
                  stage(f"{S7}/06_novel_attack_generalization.py")],
    },
    "eval_ablation": {
        "desc": "Stage 7.7-7.10: ablation studies",
        "steps": [stage(f"{S7}/07_tokenizer_ablation.py"), stage(f"{S7}/08_augmentation_ablation.py"),
                  stage(f"{S7}/09_model_size_scaling.py"), stage(f"{S7}/10_label_smoothing_ablation.py")],
    },
    "eval_interp": {
        "desc": "Stage 7.11-7.13: interpretability",
        "steps": [stage(f"{S7}/11_attention_visualization.py"), stage(f"{S7}/12_shap_analysis.py"),
                  stage(f"{S7}/13_error_analysis.py")],
    },
    "compare_all": {
        "desc": "Stage 7.14: master comparison table (runs all evals first)",
        "deps": ["eval_detection", "eval_latency", "eval_adversarial", "eval_ablation",
                 "eval_interp", "baselines_post_aug"],
        "steps": [stage(f"{S7}/14_comparison_table.py")],
    },
    "report": {"desc": "Stage 7.15: final report", "steps": [stage(f"{S7}/15_generate_report.py")]},
    "deploy_reco": {"desc": "Stage 7.16: deployment recommendation",
                    "steps": [stage(f"{S7}/16_deployment_recommendation.py")]},
    "push_hub": {
        "desc": "Stage 7.17: push artifacts to HuggingFace Hub",
        "steps": [lambda dry: run_cmd(
            stage(f"{S7}/17_push_to_hub.py", *opt("--revision", env("HF_REVISION")),
                  *opt("--only", env("HF_ONLY"))), dry)],
    },
    "push_hub_dry": {
        "desc": "Stage 7.17: preview the Hub push (uploads nothing)",
        "steps": [lambda dry: run_cmd(
            stage(f"{S7}/17_push_to_hub.py", "--dry-run", *opt("--revision", env("HF_REVISION")),
                  *opt("--only", env("HF_ONLY"))), dry)],
    },

    # ── convenience ──
    "all": {"desc": "Full pipeline end-to-end",
            "deps": ["setup", "baselines", "data_collect", "data_analyze", "data_augment_all",
                     "tokenize", "train_b_99m", "distill", "compare_all", "report"]},
    "track_b_full": {"desc": "Track B only — fastest path to results",
                     "deps": ["setup", "baselines", "data_collect", "data_augment_all",
                              "tokenize_b", "train_b_99m", "distill", "compare_all"]},
    "eval_only": {"desc": "Evaluate already-trained models",
                  "deps": ["eval_detection", "eval_latency", "eval_adversarial", "eval_ablation",
                           "eval_interp", "compare_all"]},

    # ── mlflow ──
    "ui": {"desc": "MLflow UI on :5000",
           "steps": [[PYTHON, "-m", "mlflow", "ui", "--port", "5000",
                      "--backend-store-uri", "sqlite:///mlruns/mlflow.db"]]},
    "stop_ui": {"desc": "Stop the MLflow UI", "steps": [_stop_ui]},

    # ── clean ──
    "clean_augmented": {"desc": "Remove augmented parquet", "steps": [_rmtree_globs("data/augmented/**/*.parquet")]},
    "clean_models": {"desc": "Remove trained model weights",
                     "steps": [_rmtree_globs("models/track_a/*", "models/track_b/*", "models/student/*")]},
    "clean_tokenizers": {"desc": "Remove tokenizer artifacts",
                         "steps": [_rmtree_globs("tokenizers/track_a/*", "tokenizers/track_b/*")]},
    "clean_reports": {"desc": "Remove report outputs",
                      "steps": [_rmtree_globs("reports/metrics/*", "reports/figures/*", "reports/latency/*")]},
    "clean_splits": {"desc": "Remove data splits", "steps": [_rmtree_globs("data/splits/*")]},
    "clean": {"desc": "Remove all generated artifacts",
              "deps": ["clean_augmented", "clean_models", "clean_tokenizers",
                       "clean_reports", "clean_splits"]},
}


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────
def run_cmd(cmd: list[str], dry: bool) -> None:
    print("$ " + " ".join(cmd))
    if dry:
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_task(name: str, dry: bool, done: set[str]) -> None:
    if name in done:
        return
    if name not in TASKS:
        sys.exit(f"run.py: unknown task '{name}'.  Try: python run.py --list")
    done.add(name)
    task = TASKS[name]
    for dep in task.get("deps", []):
        run_task(dep, dry, done)
    steps = task.get("steps", [])
    if steps:
        print(f"\n=== {name} — {task['desc']} ===")
    for step in steps:
        if callable(step):
            step(dry)
        else:
            run_cmd(step, dry)


def print_list() -> None:
    print("Available tasks (python run.py <task> [<task> ...]):\n")
    width = max(len(n) for n in TASKS)
    for name in TASKS:
        deps = TASKS[name].get("deps")
        suffix = f"  [deps: {', '.join(deps)}]" if deps and not TASKS[name].get("steps") else ""
        print(f"  {name:<{width}}  {TASKS[name]['desc']}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-platform task runner for ai-waf-v2 (make-free).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python run.py --list\n  python run.py train_b_99m\n  python run.py -n eval_only",
    )
    parser.add_argument("tasks", nargs="*", help="task name(s) to run, in order")
    parser.add_argument("-l", "--list", action="store_true", help="list tasks and exit")
    parser.add_argument("-n", "--dry-run", action="store_true", help="print commands without running")
    args = parser.parse_args()

    load_dotenv()

    if args.list or not args.tasks:
        print_list()
        return

    if PYTHON == sys.executable and not (ROOT / ".venv").exists():
        print("note: no .venv found — using the current interpreter "
              "(run `python run.py setup` to create one).\n")

    done: set[str] = set()
    for task in args.tasks:
        run_task(task, args.dry_run, done)


if __name__ == "__main__":
    main()
