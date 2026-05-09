"""Stage 5.4 — Compare QAT student vs PTQ teacher accuracy and latency."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    reports_dir = Path(cfg.paths.reports)
    metrics_dir = reports_dir/"metrics"
    latency_dir = reports_dir/"latency"
    # Collect results from previous stages
    files = {
        "student_val":    metrics_dir/"detection_results.json",
        "ptq_latency":    latency_dir/"ptq_comparison.json",
        "onnx_latency":   latency_dir/"onnx_bench.json",
        "latency_summary":latency_dir/"latency_summary.json",
    }
    summary: dict = {}
    for key, fpath in files.items():
        if fpath.exists():
            summary[key] = json.loads(fpath.read_text())
        else:
            log.warning(f"Missing: {fpath}")
    # Extract key comparison metrics
    comparison = {}
    if "student_val" in summary:
        for name in ["track_b_99m","student"]:
            m = summary["student_val"].get(name,{}).get("overall",{})
            if m: comparison[name] = {"f1": m.get("f1"), "auc_pr": m.get("auc_pr"), "fpr": m.get("fpr")}
    if "latency_summary" in summary:
        for model_key, data in summary["latency_summary"].get("latency_by_model",{}).items():
            results = data.get("results",[])
            bs1 = next((r for r in results if r["batch_size"]==1), None)
            if bs1: comparison.setdefault(model_key,{})["p99_ms_bs1"] = bs1["p99_ms"]
    log.info("\n=== QAT vs PTQ Comparison ===")
    for name, vals in comparison.items():
        log.info(f"  {name:30s}: {vals}")
    out = Path(cfg.paths.reports)/"metrics"/"qat_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"comparison": comparison, "raw": summary}, indent=2))
    log.info(f"QAT comparison saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
