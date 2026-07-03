"""
stages/7_evaluation/04_evasion_payloads.py
------------------------------------
Stage 7.4 — Adversarial robustness evaluation.

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
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer
from ai_waf_v2.utils.reports import report_path

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg    = load_config(args.config)

    require_inputs({
        "data/splits/test.parquet": "make data_augment_all",
        f"{cfg.model.track_b_99m.output_dir}/best_99m.pt": "run 00_train_teacher_99m.py",
    })
    if check_output(
        report_path("adversarial_summary.json", cfg.paths.reports),
        args.force, "Stage 7.4 evasion payloads"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer  = StepTimer()

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

        with timer.step(f"evaluate_{model_label}"):
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

        out_path = report_path("adversarial_summary.json", cfg.paths.reports).with_name(f"04_adversarial_{model_label}.json")
        AdversarialEvaluator.save_report(results, out_path)
        all_reports[model_label] = [
            {"tamper": r.tamper_name, "detection": r.detection_rate, "evasion": r.evasion_rate}
            for r in results
        ]

    # Summary
    report_path("adversarial_summary.json", cfg.paths.reports).write_text(
        json.dumps({**all_reports, "timings_s": timer.timings}, indent=2)
    )
    log.info(f"\nAdversarial summary saved to {report_path('adversarial_summary.json', cfg.paths.reports, mkdir=False)}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="04_evasion_payloads") as _run:
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
            mlflow.log_artifact(str(report_path("adversarial_summary.json", cfg.paths.reports, mkdir=False)))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())