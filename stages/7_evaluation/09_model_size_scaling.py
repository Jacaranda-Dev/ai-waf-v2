"""
stages/7_evaluation/09_model_size_scaling.py
-----------------------------------------
All four ablation studies in one file (dispatched by script name).
Each ablation trains a fast probe model (LR + TF-IDF) to isolate
the variable of interest without requiring full transformer training.

For full transformer ablations, use MLflow to compare existing runs:
these scripts compare saved checkpoints and report metrics.

Run:
    python stages/7_evaluation/09_model_size_scaling.py  --config config/pipeline.yaml

"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)




# ─────────────────────────────────────────────────────────
# 09 — Model size scaling
# ─────────────────────────────────────────────────────────

def model_size_scaling(cfg) -> dict:
    """Load all saved checkpoints and plot accuracy vs parameter count."""
    import json
    from ai_waf_v2.models.head import WafClassifier
    from ai_waf_v2.models.student import StudentClassifier

    timer = StepTimer()
    checkpoints = [
        ("student",    Path(cfg.model.student.output_dir)/"best_student.pt",      cfg.model.student,      True),
        ("track_b_99m",Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt",      cfg.model.track_b_99m,  False),
    ]

    reports_dir = Path(cfg.paths.reports) / "metrics"
    det_path    = reports_dir / "detection_results.json"
    if not det_path.exists():
        return {"error": "detection_results.json not found — run Stage 6.1 first"}

    det = json.loads(det_path.read_text())
    results = {}

    with timer.step("load_checkpoints"):
        for label, ckpt, arch_cfg, is_student in checkpoints:
            if not ckpt.exists(): continue
            n_params = sum(p.numel() for p in
                           (StudentClassifier.from_config(arch_cfg) if is_student
                            else WafClassifier.from_config(arch_cfg)).parameters())
            m = det.get(label, {}).get("overall", {})
            results[label] = {
                "n_params":  n_params,
                "auc_pr":    m.get("auc_pr"),
                "f1":        m.get("f1"),
                "fpr":       m.get("fpr"),
            }
            auc_pr = m.get("auc_pr")
            auc_pr_str = f"{auc_pr:.4f}" if auc_pr is not None else "N/A"
            log.info(f"  {label:20s}: params={n_params:,}  AUC-PR={auc_pr_str}")

    # Add baselines
    baselines_path = reports_dir / "baselines.json"
    if baselines_path.exists():
        baselines = json.loads(baselines_path.read_text())
        for name, data in baselines.items():
            m = data.get("overall", data.get("val_metrics", {}))
            results[name] = {"n_params": "N/A", "auc_pr": m.get("auc_pr"), "f1": m.get("f1")}

    results["timings_s"] = timer.timings
    out = reports_dir / "model_size_scaling.json"
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Model size scaling saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="09_model_size_scaling") as _run:
            metrics: dict[str, float] = {}
            for label, info in results.items():
                if not isinstance(info, dict):
                    continue
                if isinstance(info.get("n_params"), int):
                    metrics[f"{label}_n_params"] = float(info["n_params"])
                for m_key in ("auc_pr", "f1", "fpr"):
                    val = info.get(m_key)
                    if isinstance(val, (int, float)):
                        metrics[f"{label}_{m_key}"] = float(val)
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    return results



def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    require_inputs({
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "model_size_scaling.json",
        args.force, "Stage 7.9 model size scaling"
    ):
        return
    model_size_scaling(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())