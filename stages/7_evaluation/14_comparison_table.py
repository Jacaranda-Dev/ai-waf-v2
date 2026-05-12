"""
stages/7_evaluation/14_comparison_table.py
-----------------------------------------
Stage 7.6 — Aggregate metrics from all evaluation reports into a single
master comparison table (CSV + JSON).

Pulls from:
  - detection_results.json      (F1, AUC-PR, FPR, Recall)
  - latency_summary.json        (p99_ms, p99.9_ms, jitter, RPS at bs=1)
  - adversarial_summary.json    (evasion rate per model)
  - novel_attack_generalization.json (mean detection rate across novel classes)
  - memory_footprint.json        (checkpoint size MB, param count)

Output:
  - reports/metrics/master_comparison_table.csv   (human-readable)
  - reports/metrics/master_comparison_table.json  (consumed by 16_deployment_recommendation.py)

Run:
    python stages/7_evaluation/14_comparison_table.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


def _safe_load(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            log.warning(f"Failed to load {path}: {exc}")
    else:
        log.warning(f"Report file missing: {path}")
    return None


def _mean_evasion_rate(adv_data: dict | None, model_name: str) -> float | None:
    """Average evasion rate across all tamper techniques for a given model."""
    if not adv_data:
        return None
    model_results = adv_data.get(model_name, [])
    if not model_results:
        return None
    rates = [r.get("evasion", r.get("evasion_rate", 0)) for r in model_results]
    return round(sum(rates) / len(rates), 5) if rates else None


def _mean_novel_detection(novel_data: dict | None) -> float | None:
    """Mean detection rate across all novel attack classes."""
    if not novel_data:
        return None
    rates = [v.get("detection_rate", 0) for v in novel_data.values()
             if isinstance(v, dict)]
    return round(sum(rates) / len(rates), 5) if rates else None


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    reports_dir = Path(cfg.paths.reports)
    metrics_dir = reports_dir / "metrics"
    lat_dir     = reports_dir / "latency"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all report files ─────────────────────────────────────────────────
    det_data    = _safe_load(metrics_dir / "detection_results.json")        or {}
    lat_data    = _safe_load(lat_dir / "latency_summary.json")              or {}
    adv_data    = _safe_load(metrics_dir / "adversarial_summary.json")      or {}
    novel_data  = _safe_load(metrics_dir / "novel_attack_generalization.json") or {}
    mem_data    = _safe_load(metrics_dir / "memory_footprint.json")         or {}

    # Pre-compute mean novel detection (same for all model rows since the novel
    # eval currently only runs on the teacher; extend per-model when available)
    mean_novel = _mean_novel_detection(novel_data)

    rows: list[dict[str, Any]] = []

    for model_name, model_data in det_data.items():
        m = model_data.get("overall", model_data.get("val_metrics", {}))
        if not m:
            continue

        row: dict[str, Any] = {
            "Model":           model_name,
            "F1":              m.get("f1"),
            "Precision":       m.get("precision"),
            "Recall":          m.get("recall"),
            "FPR":             m.get("fpr"),
            "AUC-PR":          m.get("auc_pr"),
            "AUC-ROC":         m.get("auc_roc"),
        }

        # ── Latency (GPU bs=1) ────────────────────────────────────────────────
        lat_by_model = lat_data.get("latency_by_model", {})
        # Try GPU first, fall back to CPU
        for device_suffix in ("cuda", "cpu"):
            model_lat_key = f"{model_name}_{device_suffix}"
            model_lat = lat_by_model.get(model_lat_key, {})
            if model_lat:
                results_list = model_lat.get("results", [])
                bs1 = next((r for r in results_list if r.get("batch_size") == 1), None)
                if bs1:
                    row["p99_ms_bs1"]     = bs1.get("p99_ms")
                    row["p99_9_ms_bs1"]   = bs1.get("p99_9_ms")
                    row["jitter_std_ms"]  = bs1.get("jitter_std_ms")
                    row["RPS_bs1"]        = bs1.get("throughput_rps")
                    row["latency_device"] = device_suffix
                    slo = model_lat.get("slo", {})
                    row["SLO_p99_ok"]     = slo.get("p99_ok")
                    row["SLO_p99_9_ok"]   = slo.get("p99_9_ok")
                    row["SLO_rps_ok"]     = slo.get("rps_ok")
                    break

        # ── Adversarial robustness ────────────────────────────────────────────
        row["mean_evasion_rate"] = _mean_evasion_rate(adv_data, model_name)

        # ── Novel attack generalisation ───────────────────────────────────────
        row["mean_novel_detection"] = mean_novel

        # ── Memory footprint ──────────────────────────────────────────────────
        mem = mem_data.get(model_name, {})
        row["disk_mb"]    = mem.get("disk_mb")
        row["n_params"]   = mem.get("n_params")
        row["vram_fp16_mb"] = mem.get("vram_weights_fp16_mb")

        # ── FP diagnostic ─────────────────────────────────────────────────────
        fp = model_data.get("fp_analysis", {})
        if fp:
            row["total_fp"]        = fp.get("total_fp")
            row["fp_high_entropy"] = fp.get("category_counts", {}).get("high_entropy")
            row["fp_safe_malform"] = fp.get("category_counts", {}).get("safe_but_malformed")

        rows.append(row)

    if not rows:
        log.error("No model data found — ensure detection_results.json exists")
        return

    df = pd.DataFrame(rows)

    # ── Sort by AUC-PR descending ─────────────────────────────────────────────
    df = df.sort_values("AUC-PR", ascending=False, ignore_index=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = metrics_dir / "master_comparison_table.csv"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    log.info(f"Master comparison table (CSV) saved to {csv_path}")

    # ── Save JSON (for downstream scripts) ───────────────────────────────────
    json_path = metrics_dir / "master_comparison_table.json"
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str))
    log.info(f"Master comparison table (JSON) saved to {json_path}")

    # ── Pretty-print to log ───────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    key_cols = ["Model", "F1", "AUC-PR", "FPR", "p99_ms_bs1", "p99_9_ms_bs1",
                "jitter_std_ms", "RPS_bs1", "mean_evasion_rate", "disk_mb"]
    display_cols = [c for c in key_cols if c in df.columns]
    log.info(f"\n{df[display_cols].to_string(index=False)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())