"""
stages/7_evaluation/17_push_to_hub.py
--------------------------------------
Push all available artifacts to HuggingFace in one shot.

Individual stages push their own outputs incrementally via --save / --revision.
This script is a convenience wrapper that pushes everything that currently exists.

Auth
----
  export HF_ORG=your-org-name
  export HF_TOKEN=hf_...          # or: huggingface-cli login

Run
---
  # push everything that exists
  python stages/7_evaluation/17_push_to_hub.py

  # dry-run preview
  python stages/7_evaluation/17_push_to_hub.py --dry-run

  # tag a specific revision (e.g. pre-aug vs aug comparison)
  python stages/7_evaluation/17_push_to_hub.py --revision v1.0-aug

  # push only models or only datasets
  python stages/7_evaluation/17_push_to_hub.py --only models

  # via make
  make push_hub
  make push_hub_dry
  make push_hub HF_REVISION=v1.0-aug HF_ONLY=datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.hub import push_folder
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg  = load_config(args.config)
    only = args.only
    rev  = args.revision
    dry  = args.dry_run
    timer = StepTimer()

    if dry:
        log.info("=== DRY RUN — nothing will be uploaded ===")
    log.info(f"org={cfg.huggingface.org}  private={cfg.huggingface.private}  revision={rev}")

    pushed:  list[str] = []
    skipped: list[str] = []

    def _push(repo_key: str, folder: Path, repo_type: str, msg: str) -> None:
        with timer.step(f"push_{repo_key}"):
            ok = push_folder(cfg, folder, repo_key, repo_type, msg, rev, dry)
        (pushed if ok else skipped).append(repo_key)

    # ── Models ────────────────────────────────────────────────────────────────
    if only in ("all", "models"):
        log.info("── Models ──────────────────────────────────────────────────")
        _push("tokenizer", Path(cfg.tokenizer.track_b.output_dir),
              "model", "Upload Track B tokenizer")
        _push("teacher",   Path(cfg.model.track_b_99m.output_dir),
              "model", "Upload teacher checkpoint (99M)")
        _push("student",   Path(cfg.model.student.output_dir),
              "model", "Upload student checkpoint + ONNX (10M)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    if only in ("all", "datasets"):
        log.info("── Datasets ────────────────────────────────────────────────")
        _push("dataset_synthesis", Path(cfg.paths.data_augmented) / "synthesis",
              "dataset", "Stage 3.1: synthesized attack payloads")
        _push("dataset_framed",    Path(cfg.paths.data_augmented) / "framed",
              "dataset", "Stage 3.3: framed records (attack + benign)")
        _push("dataset_base",      Path(cfg.paths.data_splits_pre_aug),
              "dataset", "Pre-augmentation splits")
        _push("dataset_aug",       Path(cfg.paths.data_splits),
              "dataset", "Augmented splits (final)")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("── Summary ─────────────────────────────────────────────────")
    log.info(f"  Pushed  ({len(pushed)}):  {', '.join(pushed) or '—'}")
    log.info(f"  Skipped ({len(skipped)}): {', '.join(skipped) or '—'}")
    for name, elapsed in timer.timings.items():
        log.info(f"  {name}: {elapsed:.1f}s")

    try:
        import mlflow

        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="17_push_to_hub"):
            mlflow.log_params({
                "hf_org":   cfg.huggingface.org,
                "revision": rev,
                "dry_run":  dry,
                "only":     only,
                "pushed":   ",".join(pushed),
            })
            log_metrics_dict({f"pushed_{a}": 1.0 for a in pushed})
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push all WAF artifacts to HuggingFace Hub")
    p.add_argument("--config",   default="config/pipeline.yaml")
    p.add_argument("--revision", default="main",
                   help="HF revision/branch to push to (e.g. 'v1.0-aug')")
    p.add_argument("--only",     default="all", choices=["all", "models", "datasets"])
    p.add_argument("--dry-run",  action="store_true",
                   help="Preview what would be uploaded without uploading")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
