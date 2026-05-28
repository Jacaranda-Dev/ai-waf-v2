"""
stages/7_evaluation/04_evasion_payloads.py
------------------------------------
Stage 6.4 — Adversarial robustness evaluation.

Tests the trained models against:
  - All tamper scripts in TAMPER_REGISTRY
  - Evasion wordlists from config (if present)
  - Adversarial holdout split (unseen obfuscated payloads)

Reports evasion rate per tamper technique.

Run:
    python stages/7_evaluation/04_evasion_payloads.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from ai_waf_v2.data.schema import HttpRecord
from ai_waf_v2.eval.adversarial import AdversarialEvaluator, TAMPER_REGISTRY
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    # Load malicious test samples
    test_path = Path(cfg.paths.data_splits) / "test.parquet"
    if not test_path.exists():
        log.error("test.parquet not found")
        return

    table   = pq.read_table(test_path, filters=[("label", "=", 1)])
    records = [HttpRecord.from_dict(row) for row in table.to_pylist()]
    log.info(f"Loaded {len(records):,} malicious test records for adversarial eval")

    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_reports: dict[str, list] = {}

    for model_label, ckpt_path, is_student in [
        ("teacher_99m", Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt", False),
        ("student",     Path(cfg.model.student.output_dir) / "best_student.pt", True),
    ]:
        if not ckpt_path.exists():
            log.warning(f"{model_label}: checkpoint not found — skipping")
            continue

        log.info(f"\nAdversarial eval: {model_label}")

        if is_student:
            from ai_waf_v2.models.student import StudentClassifier
            model = StudentClassifier.load(ckpt_path, cfg.model.student, map_location=str(device))
        else:
            from ai_waf_v2.models.head import WafClassifier
            model = WafClassifier.load(ckpt_path, cfg.model.track_b_99m, map_location=str(device))

        evaluator = AdversarialEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            seq_len=cfg.tokenizer.seq_len,
            batch_size=64,
        )

        tampers = cfg.evaluation.adversarial.tamper_scripts or list(TAMPER_REGISTRY.keys())
        wordlists = cfg.evaluation.adversarial.evasion_wordlists

        results = evaluator.run(
            malicious_records=records[:2000],   # cap for speed
            tamper_scripts=tampers,
            evasion_wordlists=wordlists,
        )

        log.info(f"\n{'Tamper':25s}  {'Detection':>10}  {'Evasion':>10}")
        log.info("-" * 50)
        for r in results:
            log.info(
                f"{r.tamper_name:25s}  "
                f"{r.detection_rate:>10.4f}  "
                f"{r.evasion_rate:>10.4f}"
            )

        out_path = reports_dir / f"adversarial_{model_label}.json"
        AdversarialEvaluator.save_report(results, out_path)
        all_reports[model_label] = [
            {"tamper": r.tamper_name, "detection": r.detection_rate, "evasion": r.evasion_rate}
            for r in results
        ]

    # Summary
    (reports_dir / "adversarial_summary.json").write_text(
        json.dumps(all_reports, indent=2)
    )
    log.info(f"\nAdversarial summary saved to {reports_dir / 'adversarial_summary.json'}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="04_evasion_payloads"):
            mlflow.log_params({
                "n_models":         len(all_reports),
                "n_malicious_cap":  2000,
                "n_tamper_scripts": len(list(TAMPER_REGISTRY.keys())),
            })
            metrics: dict[str, float] = {}
            for model_label, tamper_list in all_reports.items():
                if tamper_list:
                    mean_det = sum(r["detection"] for r in tamper_list) / len(tamper_list)
                    mean_eva = sum(r["evasion"] for r in tamper_list) / len(tamper_list)
                    metrics[f"{model_label}_mean_detection"] = float(mean_det)
                    metrics[f"{model_label}_mean_evasion"]   = float(mean_eva)
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(reports_dir / "adversarial_summary.json"))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())