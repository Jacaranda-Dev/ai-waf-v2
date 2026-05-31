"""
stages/2_baselines/01_define_slos.py
------------------------------------
Stage 2.1 — Write SLO targets and hardware spec to reports/ so every
subsequent evaluation stage can load them without re-reading the config.

Run:
    python stages/2_baselines/01_define_slos.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output

log = get_logger(__name__)

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    require_inputs({})
    if check_output(Path(cfg.paths.reports) / "metrics" / "slos.json", args.force, "Stage 2.1 SLOs"):
        return

    slo_doc = {
        "latency": {
            "inline_waf_p99_ms":  cfg.slo.latency_inline_p99_ms,
            "offline_p99_ms":     cfg.slo.latency_offline_p99_ms,
            "throughput_min_rps": cfg.slo.throughput_min_rps,
            "notes": (
                "Inline WAF: single request at batch=1. "
                "Offline: log analysis at batch=64."
            ),
        },
        "accuracy": {
            "max_false_positive_rate": cfg.slo.max_false_positive_rate,
            "notes": (
                f"FPR <= {cfg.slo.max_false_positive_rate} means at most "
                f"{cfg.slo.max_false_positive_rate * 1000:.1f} false blocks "
                "per 1,000 legitimate requests."
            ),
        },
        "hardware": {
            "gpu":    "RTX 5090 laptop (or RTX 4090 equivalent)",
            "vram":   "24 GB",
            "cpu":    "fallback — no CUDA",
            "precision": "bf16 on GPU, fp32 on CPU",
        },
        "primary_metric": "auc_pr",
        "secondary_metrics": ["f1", "fpr", "recall", "auc_roc"],
    }

    out = Path(cfg.paths.reports) / "metrics" / "slos.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slo_doc, indent=2))
    log.info(f"SLO document written to {out}")

    for k, v in slo_doc["latency"].items():
        if k != "notes":
            log.info(f"  latency.{k} = {v}")
    log.info(f"  accuracy.max_fpr = {slo_doc['accuracy']['max_false_positive_rate']}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_define_slos"):
            mlflow.log_params({
                "latency_inline_p99_ms":  cfg.slo.latency_inline_p99_ms,
                "latency_offline_p99_ms": cfg.slo.latency_offline_p99_ms,
                "throughput_min_rps":     cfg.slo.throughput_min_rps,
                "max_false_positive_rate": cfg.slo.max_false_positive_rate,
                "primary_metric":         slo_doc["primary_metric"],
            })
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())