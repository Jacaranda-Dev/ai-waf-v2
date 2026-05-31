"""
stages/7_evaluation/15_generate_report.py
-----------------------------------------
Stage 7.7 — Generate the final consolidated evaluation report.

Consolidates all Stage 7 JSON outputs into a single structured document:
  - Summary table (model × key metric)
  - Detection efficacy
  - Latency / throughput (incl. tail latency p99.9 and jitter)
  - Memory footprint
  - Adversarial robustness
  - Obfuscation robustness (chained tampers)
  - Novel attack generalisation (with confidence intervals)
  - Tokenizer & augmentation ablations
  - Model size scaling
  - Label smoothing ablation
  - Attention visualisation and SHAP analysis (summaries)
  - Error analysis (FP/FN distribution)
  - FP diagnostic clustering
  - Deployment recommendation (from script 16 if available)

Output:
  - reports/final_evaluation_report.json

Run:
    python stages/7_evaluation/15_generate_report.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


SECTION_MANIFEST: dict[str, str] = {
    "detection_efficacy":           "metrics/detection_results.json",
    "latency_throughput":           "latency/latency_summary.json",
    "memory_footprint":             "metrics/memory_footprint.json",
    "adversarial_robustness":       "metrics/adversarial_summary.json",
    "obfuscation_robustness":       "metrics/obfuscation_robustness.json",
    "novel_attack_generalization":  "metrics/novel_attack_generalization.json",
    "tokenizer_ablation":           "metrics/tokenizer_ablation.json",
    "augmentation_ablation":        "metrics/augmentation_ablation.json",
    "model_size_scaling":           "metrics/model_size_scaling.json",
    "label_smoothing_ablation":     "metrics/label_smoothing_ablation.json",
    "attention_visualization":      "metrics/attention_visualization.json",
    "shap_analysis":                "metrics/shap_analysis.json",
    "error_analysis":               "metrics/error_analysis.json",
    "master_comparison_table":      "metrics/master_comparison_table.json",
    "deployment_recommendation":    "metrics/deployment_recommendation.json",
}


def _safe_load(path: Path) -> Any | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            log.warning(f"  Could not load {path.name}: {exc}")
    return None


def _build_summary(det_data: dict | None, lat_data: dict | None,
                   adv_data: dict | None) -> dict:
    """
    Construct a concise summary dict:
        model_name → {f1, auc_pr, fpr, p99_ms (gpu, bs=1), mean_evasion_rate}
    """
    summary: dict[str, dict] = {}
    if not det_data:
        return summary

    lat_by_model = (lat_data or {}).get("latency_by_model", {})

    for model_name, model_data in det_data.items():
        m = model_data.get("overall", model_data.get("val_metrics", {}))
        if not m:
            continue

        entry: dict[str, Any] = {
            "f1":      m.get("f1"),
            "auc_pr":  m.get("auc_pr"),
            "fpr":     m.get("fpr"),
            "recall":  m.get("recall"),
        }

        # Latency: first available (prefer cuda)
        for suffix in ("cuda", "cpu"):
            lat_key = f"{model_name}_{suffix}"
            lat_results = lat_by_model.get(lat_key, {}).get("results", [])
            bs1 = next((r for r in lat_results if r.get("batch_size") == 1), None)
            if bs1:
                entry["p99_ms"]     = bs1.get("p99_ms")
                entry["p99_9_ms"]   = bs1.get("p99_9_ms")
                entry["jitter_ms"]  = bs1.get("jitter_std_ms")
                entry["rps"]        = bs1.get("throughput_rps")
                entry["latency_device"] = suffix
                break

        # Adversarial: mean evasion rate
        adv_results = (adv_data or {}).get(model_name, [])
        if adv_results:
            evasion_rates = [r.get("evasion", r.get("evasion_rate", 0))
                             for r in adv_results]
            entry["mean_evasion_rate"] = round(
                sum(evasion_rates) / len(evasion_rates), 5
            )

        # SLO pass/fail
        lat_model_key = f"{model_name}_cuda"
        slo = lat_by_model.get(lat_model_key, {}).get("slo", {})
        if slo:
            entry["slo"] = {
                "p99_ok":   slo.get("p99_ok"),
                "p99_9_ok": slo.get("p99_9_ok"),
                "rps_ok":   slo.get("rps_ok"),
            }

        summary[model_name] = entry

    return summary


def _latency_tail_digest(lat_data: dict | None) -> dict:
    """
    Extract p99.9 and jitter highlights so the report surface area around
    tail latency (critique §3) is explicit.
    """
    if not lat_data:
        return {}
    digest: dict = {"pcie_overhead_ms": lat_data.get("pcie_overhead_ms", {})}
    for model_key, model_val in lat_data.get("latency_by_model", {}).items():
        results = model_val.get("results", [])
        bs1 = next((r for r in results if r.get("batch_size") == 1), None)
        if bs1:
            digest[model_key] = {
                "p99_ms":     bs1.get("p99_ms"),
                "p99_9_ms":   bs1.get("p99_9_ms"),
                "jitter_ms":  bs1.get("jitter_std_ms"),
            }
    return digest


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)

    require_inputs({
        f"{Path(cfg.paths.reports) / 'metrics' / 'master_comparison_table.json'}": "run 14_comparison_table.py",
    })
    if check_output(
        Path(cfg.paths.reports) / "final_evaluation_report.json",
        args.force, "Stage 7.15 generate report"
    ):
        return

    reports_dir = Path(cfg.paths.reports)

    log.info("Assembling final evaluation report...")
    timer = StepTimer()

    # ── Load all sections ─────────────────────────────────────────────────────
    sections: dict[str, Any] = {}
    with timer.step("load_sections"):
        for section_name, rel_path in SECTION_MANIFEST.items():
            full_path = reports_dir / rel_path
            data      = _safe_load(full_path)
            if data is not None:
                sections[section_name] = data
                log.info(f"  ✓ {section_name}")
            else:
                log.warning(f"  ✗ {section_name} — missing ({full_path})")

    # ── Build summary ─────────────────────────────────────────────────────────
    with timer.step("build_report"):
        summary = _build_summary(
            sections.get("detection_efficacy"),
            sections.get("latency_throughput"),
            sections.get("adversarial_robustness"),
        )

        tail_digest = _latency_tail_digest(sections.get("latency_throughput"))

        # ── Novel attack CI summary ───────────────────────────────────────────────
        novel_ci_summary: dict = {}
        for attack, stats in (sections.get("novel_attack_generalization") or {}).items():
            if isinstance(stats, dict):
                novel_ci_summary[attack] = {
                    "detection_rate": stats.get("detection_rate"),
                    "ci_95":          [stats.get("ci_95_low"), stats.get("ci_95_high")],
                    "n_unique":       stats.get("n_unique"),
                }

        # ── FP clustering summary (from detection_efficacy) ───────────────────────
        fp_cluster_summary: dict = {}
        for model_name, model_data in (sections.get("detection_efficacy") or {}).items():
            fp_analysis = model_data.get("fp_analysis") if isinstance(model_data, dict) else None
            if fp_analysis:
                fp_cluster_summary[model_name] = {
                    "total_fp":       fp_analysis.get("total_fp"),
                    "category_counts": fp_analysis.get("category_counts"),
                    "n_clusters":     len(fp_analysis.get("kmeans_clusters", [])),
                }

        # ── Tokenizer correction summary ─────────────────────────────────────────
        tok_correction: dict = {}
        tok_abl = sections.get("tokenizer_ablation", {})
        if "transformer_correction" in tok_abl:
            tok_correction = tok_abl["transformer_correction"]

        # ── Assemble final report ─────────────────────────────────────────────────
        final_report: dict[str, Any] = {
            "report_metadata": {
                "generated_at":  datetime.now(timezone.utc).isoformat(),
                "config":        str(args.config),
                "sections_found": list(sections.keys()),
                "sections_missing": [k for k in SECTION_MANIFEST if k not in sections],
            },
            "summary": summary,
            "tail_latency_digest": tail_digest,
            "novel_attack_ci_summary": novel_ci_summary,
            "fp_diagnostic_summary": fp_cluster_summary,
            "tokenizer_anchor_correction": tok_correction,
            **sections,   # full data for each section
        }

        out = reports_dir / "final_evaluation_report.json"
        out.write_text(json.dumps(final_report, indent=2, default=str))
        log.info(f"\nFinal evaluation report generated: {out}")
        log.info(f"  Sections included: {len(sections)}/{len(SECTION_MANIFEST)}")
        log.info(f"  Models in summary: {list(summary.keys())}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="15_generate_report"):
            mlflow.log_params({
                "sections_found":   len(sections),
                "sections_total":   len(SECTION_MANIFEST),
                "models_in_summary": len(summary),
            })
            metrics: dict[str, float] = {
                "n_sections_found":   float(len(sections)),
                "n_sections_missing": float(len(SECTION_MANIFEST) - len(sections)),
            }
            for model_name, model_summary in summary.items():
                for key in ("f1", "auc_pr", "fpr", "recall"):
                    val = model_summary.get(key)
                    if isinstance(val, (int, float)):
                        metrics[f"{model_name}_{key}"] = float(val)
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
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