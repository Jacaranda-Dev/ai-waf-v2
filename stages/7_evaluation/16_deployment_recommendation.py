"""
stages/7_evaluation/16_deployment_recommendation.py
----------------------------------------------------
Stage 7.8 — Deployment Decision Matrix.

Critique §6 — Implements the weighted scoring algorithm:

    score(model) =
        0.5 × Normalized(AUC-PR)
      + 0.3 × Normalized(latency_score)    # 1 if p99 ≤ SLO, degrades linearly above
      + 0.2 × Normalized(1 - mean_evasion_rate)

Reads from:
  - master_comparison_table.json   (aggregated by 14_comparison_table.py)
  - latency_summary.json           (SLO targets)

Outputs:
  - reports/metrics/deployment_recommendation.json

Run:
    python stages/7_evaluation/16_deployment_recommendation.py \
        --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

# ── Scoring weights (sum to 1.0) ─────────────────────────────────────────────
W_EFFICACY   = 0.5   # AUC-PR
W_LATENCY    = 0.3   # latency score (p99 vs SLO)
W_ROBUSTNESS = 0.2   # 1 − mean_evasion_rate


def _Normalize(values: list[float | None]) -> list[float]:
    """Min-max Normalize a list, treating None as 0."""
    cleaned = [v if v is not None else 0.0 for v in values]
    lo, hi  = min(cleaned), max(cleaned)
    if hi == lo:
        return [1.0 if v > 0 else 0.0 for v in cleaned]
    return [(v - lo) / (hi - lo) for v in cleaned]


def _latency_score(p99_ms: float | None, slo_ms: float) -> float:
    """
    Returns 1.0 if p99 ≤ SLO, then degrades linearly to 0 at 2× SLO.
    Models missing latency data receive 0.
    """
    if p99_ms is None:
        return 0.0
    if p99_ms <= slo_ms:
        return 1.0
    overshoot = (p99_ms - slo_ms) / slo_ms   # 0 at SLO, 1 at 2×SLO
    return max(0.0, 1.0 - overshoot)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)

    require_inputs({
        f"{Path(cfg.paths.reports) / 'metrics' / 'master_comparison_table.json'}": "run 14_comparison_table.py",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "deployment_recommendation.json",
        args.force, "Stage 7.16 deployment recommendation"
    ):
        return

    reports_dir = Path(cfg.paths.reports)
    metrics_dir = reports_dir / "metrics"
    lat_dir     = reports_dir / "latency"

    # ── Load inputs ───────────────────────────────────────────────────────────
    table_path = metrics_dir / "master_comparison_table.json"
    lat_path   = lat_dir / "latency_summary.json"

    if not table_path.exists():
        log.error(f"master_comparison_table.json not found — run 14_comparison_table.py first")
        return

    table: list[dict[str, Any]] = json.loads(table_path.read_text())
    lat_data: dict              = json.loads(lat_path.read_text()) if lat_path.exists() else {}

    slo_p99_ms      = lat_data.get("slo_targets", {}).get("inline_p99_ms",
                                   cfg.slo.latency_inline_p99_ms)
    slo_rps         = lat_data.get("slo_targets", {}).get("throughput_rps",
                                   cfg.slo.throughput_min_rps)

    if not table:
        log.error("master_comparison_table.json is empty")
        return

    timer = StepTimer()

    # ── Extract raw scores ────────────────────────────────────────────────────
    with timer.step("score_models"):
        models   = [r["Model"] for r in table]
        auc_prs  = [r.get("AUC-PR") for r in table]
        p99s     = [r.get("p99_ms_bs1") for r in table]
        evasions = [r.get("mean_evasion_rate") for r in table]

        # Robustness = 1 − evasion_rate  (higher is better)
        robustness = [
            (1.0 - e) if e is not None else None
            for e in evasions
        ]

        # Latency score per model (not Normalized — already in [0,1])
        lat_scores_raw = [_latency_score(p99, slo_p99_ms) for p99 in p99s]

        # Normalize AUC-PR and robustness
        norm_auc_pr     = _Normalize(auc_prs)
        norm_latency    = _Normalize(lat_scores_raw)   # relative ranking on top of [0,1]
        norm_robustness = _Normalize(robustness)

        # ── Weighted composite score ──────────────────────────────────────────────
        scored: list[dict[str, Any]] = []
        for i, (model, row) in enumerate(zip(models, table)):
            composite = (
                W_EFFICACY   * norm_auc_pr[i]
                + W_LATENCY    * norm_latency[i]
                + W_ROBUSTNESS * norm_robustness[i]
            )
            entry: dict[str, Any] = {
                "model":                  model,
                "composite_score":        round(composite, 5),
                "component_scores": {
                    "efficacy_raw":       auc_prs[i],
                    "efficacy_norm":      round(norm_auc_pr[i], 5),
                    "latency_p99_ms":     p99s[i],
                    "latency_score_raw":  round(lat_scores_raw[i], 5),
                    "latency_norm":       round(norm_latency[i], 5),
                    "mean_evasion_rate":  evasions[i],
                    "robustness_raw":     robustness[i],
                    "robustness_norm":    round(norm_robustness[i], 5),
                },
                "slo_checks": {
                    "p99_ms":   p99s[i],
                    "p99_ok":   (p99s[i] is not None and p99s[i] <= slo_p99_ms),
                    "rps":      row.get("RPS_bs1"),
                    "rps_ok":   (row.get("RPS_bs1") is not None
                                 and row["RPS_bs1"] >= slo_rps),
                    "p99_9_ms": row.get("p99_9_ms_bs1"),
                    "p99_9_ok": (row.get("p99_9_ms_bs1") is not None
                                 and row["p99_9_ms_bs1"] <= slo_p99_ms * 1.5),
                },
                "fp_diagnostics": {
                    "total_fp":         row.get("total_fp"),
                    "high_entropy_fp":  row.get("fp_high_entropy"),
                    "safe_malform_fp":  row.get("fp_safe_malform"),
                },
                "model_metadata": {
                    "disk_mb":     row.get("disk_mb"),
                    "n_params":    row.get("n_params"),
                    "vram_fp16_mb": row.get("vram_fp16_mb"),
                },
            }
            scored.append(entry)

        # Sort descending by composite score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

    winner        = scored[0]
    runner_up     = scored[1] if len(scored) > 1 else None
    winner_reason = _generate_reasoning(winner, runner_up, slo_p99_ms)

    # ── Output ────────────────────────────────────────────────────────────────
    recommendation: dict[str, Any] = {
        "recommendation": {
            "selected_model":   winner["model"],
            "composite_score":  winner["composite_score"],
            "reasoning":        winner_reason,
        },
        "scoring_weights": {
            "detection_efficacy_auc_pr": W_EFFICACY,
            "operational_latency_p99":   W_LATENCY,
            "adversarial_robustness":    W_ROBUSTNESS,
        },
        "slo_targets": {
            "inline_p99_ms":         slo_p99_ms,
            "tail_p99_9_budget_ms":  slo_p99_ms * 1.5,
            "throughput_min_rps":    slo_rps,
        },
        "all_models_ranked": scored,
    }

    out = metrics_dir / "deployment_recommendation.json"
    out.write_text(json.dumps(recommendation, indent=2, default=str))

    # ── Console summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  DEPLOYMENT DECISION MATRIX")
    log.info("=" * 60)
    log.info(f"  {'Model':25s}  {'Score':>7}  {'AUC-PR':>7}  {'p99 ms':>8}  {'Evasion':>8}")
    log.info("-" * 60)
    for entry in scored:
        cs = entry["component_scores"]
        log.info(
            f"  {entry['model']:25s}  "
            f"{entry['composite_score']:>7.4f}  "
            f"{cs['efficacy_raw'] or 0:>7.4f}  "
            f"{cs['latency_p99_ms'] or 0:>8.1f}  "
            f"{cs['mean_evasion_rate'] or 0:>8.4f}"
        )
    log.info("=" * 60)
    log.info(f"  ✓ RECOMMENDED: {winner['model']} (score={winner['composite_score']:.4f})")
    log.info(f"  {winner_reason}")
    log.info(f"\nDeployment recommendation saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="16_deployment_recommendation"):
            mlflow.log_params({
                "w_efficacy":          W_EFFICACY,
                "w_latency":           W_LATENCY,
                "w_robustness":        W_ROBUSTNESS,
                "slo_p99_ms":          slo_p99_ms,
                "slo_rps":             slo_rps,
                "selected_model":      winner["model"],
                "n_models_ranked":     len(scored),
            })
            metrics: dict[str, float] = {
                "winner_composite_score": float(winner["composite_score"]),
            }
            if runner_up:
                metrics["runner_up_composite_score"] = float(runner_up["composite_score"])
                metrics["winner_runner_up_gap"] = float(
                    winner["composite_score"] - runner_up["composite_score"]
                )
            cs = winner["component_scores"]
            if cs.get("efficacy_raw") is not None:
                metrics["winner_auc_pr"]        = float(cs["efficacy_raw"])
            if cs.get("latency_p99_ms") is not None:
                metrics["winner_p99_ms"]        = float(cs["latency_p99_ms"])
            if cs.get("mean_evasion_rate") is not None:
                metrics["winner_evasion_rate"]  = float(cs["mean_evasion_rate"])
            slo_checks = winner["slo_checks"]
            metrics["winner_p99_ok"]  = float(slo_checks.get("p99_ok", False))
            metrics["winner_rps_ok"]  = float(slo_checks.get("rps_ok", False))
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def _generate_reasoning(
    winner: dict[str, Any],
    runner_up: dict | None,
    slo_ms: float,
) -> str:
    """Produce a brief human-readable justification for the selection."""
    cs   = winner["component_scores"]
    slo  = winner["slo_checks"]
    parts: list[str] = []

    parts.append(
        f"Selected {winner['model']} with composite score {winner['composite_score']:.4f} "
        f"(weights: efficacy={W_EFFICACY}, latency={W_LATENCY}, robustness={W_ROBUSTNESS})."
    )

    if cs["efficacy_raw"] is not None:
        parts.append(f"AUC-PR={cs['efficacy_raw']:.4f}.")

    if slo["p99_ok"]:
        parts.append(f"p99 latency {slo['p99_ms']:.1f} ms is within SLO ({slo_ms} ms).")
    else:
        parts.append(
            f"WARNING: p99 latency {slo['p99_ms']} ms exceeds SLO ({slo_ms} ms); "
            "consider optimisation before production deployment."
        )

    if slo.get("p99_9_ok") is False:
        parts.append(
            f"Tail latency (p99.9={slo['p99_9_ms']} ms) exceeds 1.5× SLO budget — "
            "investigate GC pauses or kernel scheduling jitter."
        )

    if cs["mean_evasion_rate"] is not None:
        parts.append(f"Mean adversarial evasion rate: {cs['mean_evasion_rate']:.4f}.")

    if runner_up:
        gap = winner["composite_score"] - runner_up["composite_score"]
        parts.append(
            f"Runner-up: {runner_up['model']} "
            f"(score={runner_up['composite_score']:.4f}, gap={gap:.4f})."
        )

    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())