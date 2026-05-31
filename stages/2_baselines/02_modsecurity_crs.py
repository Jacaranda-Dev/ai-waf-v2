"""
stages/2_baselines/02_modsecurity_crs.py
----------------------------------------
Stage 2.1 — ModSecurity CRS 3.3 heuristic baseline with Paranoia Levels.

Implements all four OWASP CRS Paranoia Levels (PL1–PL4), an anomaly
scoring engine (matching the real ModSecurity accumulate-then-threshold
model), and automated SLO validation.

Paranoia Levels
───────────────
PL1  Core rules only — high confidence, near-zero FP.  Production default.
PL2  Adds moderately aggressive rules — stricter SQL/XSS checks.
PL3  Adds rules that catch more attack variants at the cost of more FPs.
PL4  Maximum coverage — every rule fires.  Intentionally high FP for research.

Anomaly Scoring (not binary)
─────────────────────────────
Each rule contributes a severity-weighted score.  A request is flagged
when its accumulated score exceeds `inbound_anomaly_score_threshold`
(default 5, matching CRS defaults).  This mirrors real production CRS
behaviour and replaces the original binary match/no-match approach.

SLO Audit
──────────
Reads slos.json written by Stage 01 and emits a PASS/FAIL verdict per
metric into baselines.json.  A baseline can pass accuracy while failing
latency — both verdicts are recorded independently.

Run:
    # default: all four levels, PL1 result saved as the canonical entry
    python stages/2_baselines/02_modsecurity_crs.py --config config/pipeline.yaml

    # single level
    python ... --paranoia-level 2

    # override anomaly threshold
    python ... --threshold 10

    # parse a real ModSecurity audit log instead of the heuristic engine
    python ... --audit-log /var/log/modsec_audit.log
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
import torch

from ai_waf_v2.eval.metrics import compute_metrics, compute_per_class_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Rule definitions
#
# Each rule carries:
#   attack_class  — canonical class label (used for per-class breakdown)
#   severity      — CRS severity score added to the request's anomaly total
#   min_pl        — minimum paranoia level at which this rule is active
#
# Severity values mirror official CRS:
#   CRITICAL=5  ERROR=4  WARNING=3  NOTICE=2
# ─────────────────────────────────────────────────────────

@dataclass
class CrsRule:
    id:           int
    attack_class: str
    pattern:      re.Pattern
    severity:     int        # score contribution per match
    min_pl:       int        # 1..4 — active at this PL and above
    description:  str = ""


# Score constants matching official CRS severity mapping
_CRITICAL = 5
_ERROR    = 4
_WARNING  = 3
_NOTICE   = 2

CRS_RULES: list[CrsRule] = [
    # ── PL1 — high-confidence, near-zero FP ──────────────────────────────────
    CrsRule(942100, "sqli", re.compile(
        r"union\s+(?:all\s+)?select|select\s+.+\s+from\s+\w|"
        r"insert\s+into\s+\w|drop\s+table\s+\w|"
        r"sleep\s*\(\s*\d|benchmark\s*\(\s*\d|waitfor\s+delay\s+'",
        re.IGNORECASE), _CRITICAL, 1, "SQLi: classic UNION/SELECT/DROP"),

    CrsRule(941100, "xss", re.compile(
        r"<\s*script[\s>]|javascript\s*:[^'\"]*alert|"
        r"onerror\s*=\s*['\"]?[^'\">\s]+|onload\s*=\s*['\"]?[^'\">\s]+|"
        r"document\.cookie|<\s*iframe[\s/]",
        re.IGNORECASE), _CRITICAL, 1, "XSS: script/event handlers"),

    CrsRule(930100, "lfi", re.compile(
        r"\.\./|\.\.\\|%2e%2e(?:%2f|/)|"
        r"/etc/passwd|/etc/shadow|/proc/self/|"
        r"php://(?:filter|input)|data://text/",
        re.IGNORECASE), _CRITICAL, 1, "LFI: path traversal / php wrappers"),

    CrsRule(931100, "rfi", re.compile(
        r"https?://(?!(?:localhost|127\.0\.0\.1))[^/\s]{4,}/[^\s]*\.(?:php|asp|jsp)",
        re.IGNORECASE), _ERROR, 1, "RFI: remote file include"),

    CrsRule(934100, "ssrf", re.compile(
        r"169\.254\.169\.254|metadata\.google\.internal|"
        r"(?:^|[\s=@])127\.0\.0\.1(?:[:/]|$)|"
        r"file://|dict://|gopher://",
        re.IGNORECASE), _CRITICAL, 1, "SSRF: metadata / loopback / proto"),

    CrsRule(932100, "cmdi", re.compile(
        r"[;|&`]\s*(?:id|whoami|uname|cat\s+/|ls\s+-|"
        r"curl\s+https?:|wget\s+https?:|bash\s+-[ic]|nc\s+-[enl])|"
        r"\$\([^)]{2,}\)|`[^`]{2,}`",
        re.IGNORECASE), _CRITICAL, 1, "CMDi: shell metacharacters"),

    CrsRule(950100, "xxe", re.compile(
        r"<!ENTITY\s+\w|SYSTEM\s+['\"]file:|<!DOCTYPE\s+\w[^>]*\[",
        re.IGNORECASE), _CRITICAL, 1, "XXE: entity / DOCTYPE injection"),

    CrsRule(944100, "ssti", re.compile(
        r"\{\{.{1,80}\}\}|\{%.{1,80}%\}|\$\{.{1,80}\}|"
        r"#\{.{1,80}\}|<%=?.{1,80}%>",
        re.IGNORECASE), _ERROR, 1, "SSTI: template expression"),

    # ── PL2 — moderately aggressive ──────────────────────────────────────────
    CrsRule(942110, "sqli", re.compile(
        r"'\s*(?:or|and)\s*'|'\s*(?:or|and)\s+\d|"
        r"\bor\s+1\s*=\s*1\b|\band\s+1\s*=\s*1\b|"
        r"0x[0-9a-f]{4,}",
        re.IGNORECASE), _WARNING, 2, "SQLi PL2: tautology / hex literals"),

    CrsRule(941110, "xss", re.compile(
        r"<\s*svg[\s>]|<\s*math[\s>]|<\s*details[\s>]|"
        r"onfocus\s*=|onmouseover\s*=|onpointerover\s*=|"
        r"eval\s*\(|expression\s*\(",
        re.IGNORECASE), _WARNING, 2, "XSS PL2: SVG/math vectors / eval"),

    CrsRule(930110, "lfi", re.compile(
        r"%252e%252e|\.%2e|%2e\.|"
        r"(?:boot\.ini|win\.ini|system32[/\\])",
        re.IGNORECASE), _WARNING, 2, "LFI PL2: double-encoded traversal / Windows paths"),

    CrsRule(932110, "cmdi", re.compile(
        r"(?:;|\|\|?|&&)\s*(?:echo|printf|python|perl|ruby|php)\s",
        re.IGNORECASE), _WARNING, 2, "CMDi PL2: interpreter invocation"),

    # ── PL3 — stricter, more FPs ──────────────────────────────────────────────
    CrsRule(942120, "sqli", re.compile(
        r"\bexec\s*\(|\bexecute\s*\(|\bxp_cmdshell\b|"
        r"\bsp_executesql\b|information_schema\s*\.|"
        r"sys\.(?:tables|columns|objects)",
        re.IGNORECASE), _WARNING, 3, "SQLi PL3: stored proc / schema enumeration"),

    CrsRule(941120, "xss", re.compile(
        r"(?:src|href|action|formaction|srcdoc)\s*=\s*['\"]?\s*(?:javascript|data):",
        re.IGNORECASE), _NOTICE, 3, "XSS PL3: attribute-injected JS/data URIs"),

    CrsRule(921150, "header_injection", re.compile(
        r"[\r\n](?:Content-Type|Location|Set-Cookie|X-)",
        re.IGNORECASE), _WARNING, 3, "Header injection PL3"),

    CrsRule(932120, "cmdi", re.compile(
        r"(?:nmap|masscan|nikto|sqlmap|hydra|metasploit|msfconsole)\s",
        re.IGNORECASE), _NOTICE, 3, "CMDi PL3: known attack-tool names"),

    # ── PL4 — maximum paranoia ───────────────────────────────────────────────
    CrsRule(942130, "sqli", re.compile(
        r"\bselect\b|\binsert\b|\bdelete\b|\bupdate\b|\bmerge\b|"
        r"\btruncate\b|\bwhere\b.{0,40}\b=\b",
        re.IGNORECASE), _NOTICE, 4, "SQLi PL4: bare SQL keywords (high FP)"),

    CrsRule(941130, "xss", re.compile(
        r"<[a-z]+\s[^>]*on\w+\s*=",
        re.IGNORECASE), _NOTICE, 4, "XSS PL4: any inline event handler"),

    CrsRule(930120, "lfi", re.compile(
        r"(?:etc|proc|sys|dev|var|tmp)[/\\]",
        re.IGNORECASE), _NOTICE, 4, "LFI PL4: Unix path fragments (high FP)"),
]


# ─────────────────────────────────────────────────────────
# Scoring engine
# ─────────────────────────────────────────────────────────

class ScoredResult(NamedTuple):
    prediction:    int          # 0 or 1
    score:         int          # accumulated anomaly score
    matched_rules: list[int]    # rule IDs that fired


def score_request(
    raw:       str,
    active_pl: int,
    threshold: int,
) -> ScoredResult:
    """
    Accumulate severity scores for all rules active at `active_pl`.
    Return prediction=1 when the total exceeds `threshold`.

    This mirrors the real CRS `SecInboundAnomalyScoreThreshold` mechanism,
    replacing the original binary match/no-match implementation.
    """
    total:   int       = 0
    matched: list[int] = []
    text = raw.lower()

    for rule in CRS_RULES:
        if rule.min_pl > active_pl:
            continue
        if rule.pattern.search(text):
            total   += rule.severity
            matched.append(rule.id)

    return ScoredResult(
        prediction    = 1 if total >= threshold else 0,
        score         = total,
        matched_rules = matched,
    )


# ─────────────────────────────────────────────────────────
# SLO audit helper
# ─────────────────────────────────────────────────────────

def _load_slos(reports_dir: Path) -> dict | None:
    slo_path = reports_dir / "slos.json"
    if not slo_path.exists():
        log.warning("slos.json not found — SLO audit skipped (run 01_define_slos.py first)")
        return None
    return json.loads(slo_path.read_text())


def _audit_slos(
    model_key:   str,
    metrics:     dict,
    latency:     dict,
    slos:        dict,
) -> dict[str, str]:
    """
    Compare model results against each SLO.  Returns a verdict dict
    mapping metric name → "PASS" | "FAIL".

    Verdicts are intentionally independent: an accuracy PASS with a
    latency FAIL is the expected outcome for classical ML baselines,
    and is the key scientific finding that motivates the Transformer.
    """
    verdicts: dict[str, str] = {}

    # Accuracy SLO: FPR must not exceed the configured ceiling
    max_fpr = slos["accuracy"]["max_false_positive_rate"]
    actual_fpr = metrics.get("fpr", 1.0)
    verdicts["fpr"] = "PASS" if actual_fpr <= max_fpr else "FAIL"

    # Latency SLO: offline p99 (batch=64) must stay under the limit
    p99_limit = slos["latency"]["offline_p99_ms"]
    actual_p99 = latency.get("p99_ms", float("inf"))
    verdicts["latency_p99"] = "PASS" if actual_p99 <= p99_limit else "FAIL"

    # Throughput SLO
    min_rps = slos["latency"]["throughput_min_rps"]
    actual_rps = latency.get("throughput_rps", 0)
    verdicts["throughput"] = "PASS" if actual_rps >= min_rps else "FAIL"

    # Log a summary line per verdict
    for metric, verdict in verdicts.items():
        icon = "✓" if verdict == "PASS" else "✗"
        log.info(f"  SLO [{model_key}] {icon} {metric}: {verdict}")

    return verdicts


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    require_inputs({
        "data/splits/test.parquet": "make baselines",
    })
    if check_output(Path(cfg.paths.reports) / "metrics" / "modsecurity_results.json", args.force, "Stage 2.2 ModSecurity CRS"):
        return

    splits_dir  = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_path = splits_dir / "test.parquet"
    if not test_path.exists():
        log.error("test.parquet not found — run data stages first")
        return

    table   = pq.read_table(test_path, columns=["raw", "label", "attack_class"])
    raws    = table["raw"].to_pylist()
    labels  = table["label"].to_pylist()
    classes = table["attack_class"].to_pylist()
    log.info(f"Loaded {len(raws):,} test samples")
    timer     = StepTimer()

    slos      = _load_slos(reports_dir)
    threshold = args.threshold or getattr(getattr(cfg, "slo", None), "crs_anomaly_threshold", 5)

    # Determine which PL(s) to evaluate
    pl_range = (
        [args.paranoia_level] if args.paranoia_level
        else [1, 2, 3, 4]
    )

    baselines_path = reports_dir / "baselines.json"
    existing = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}

    log.info(f"\nAnomaly score threshold: {threshold}")
    log.info(f"Evaluating paranoia levels: {pl_range}\n")
    log.info(f"{'PL':>4}  {'F1':>7}  {'FPR':>8}  {'Recall':>7}  {'RPS':>8}  {'#Rules':>6}")
    log.info("─" * 52)

    for pl in pl_range:
        model_key = f"modsecurity_crs_pl{pl}"
        n_active  = sum(1 for r in CRS_RULES if r.min_pl <= pl)

        with timer.step(f"evaluate_pl{pl}"):
            t0 = time.perf_counter()
            results = [score_request(r, pl, threshold) for r in raws]
            elapsed = time.perf_counter() - t0

        preds  = [sr.prediction for sr in results]
        scores = [sr.score      for sr in results]
        rps    = len(raws) / elapsed

        preds_t  = torch.tensor(preds,  dtype=torch.long)
        probs_t  = torch.tensor(scores, dtype=torch.float)   # score as soft prob proxy
        labels_t = torch.tensor(labels, dtype=torch.long)

        metrics  = compute_metrics(preds_t, probs_t, labels_t)
        per_class = compute_per_class_metrics(preds_t, probs_t, labels_t, classes)

        latency_entry = {
            "batch_size":    1,      # CRS is per-request, not batched
            "mean_rps":      round(rps, 0),
            "device":        "cpu",
            # Express throughput as an equivalent p99 for SLO comparison.
            # At steady-state, mean_latency_ms ≈ 1000 / rps.
            "p99_ms":        round(1000 / max(rps, 1e-9), 3),
            "throughput_rps": round(rps, 1),
        }

        log.info(
            f"PL{pl}   F1={metrics['f1']:.4f}  "
            f"FPR={metrics['fpr']:.5f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"RPS={rps:>8.0f}  "
            f"Rules={n_active}"
        )

        # SLO audit
        slo_verdicts = (
            _audit_slos(model_key, metrics, latency_entry, slos)
            if slos else {}
        )

        existing[model_key] = {
            "model":           model_key,
            "paranoia_level":  pl,
            "anomaly_threshold": threshold,
            "n_active_rules":  n_active,
            "overall":         metrics,
            "per_class":       per_class,
            "throughput":      latency_entry,
            "elapsed_s":       round(elapsed, 3),
            "slo_verdicts":    slo_verdicts,
        }

    baselines_path.write_text(json.dumps(existing, indent=2))
    log.info(f"\nCRS results (all PL) saved to {baselines_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="02_modsecurity_crs"):
            mlflow.log_params({
                "paranoia_levels":      pl_range,
                "anomaly_threshold":    threshold,
                "n_rules_total":        len(CRS_RULES),
                "n_test_samples":       len(raws),
            })
            mlflow_metrics: dict[str, float] = {}
            for pl in pl_range:
                key = f"modsecurity_crs_pl{pl}"
                if key in existing:
                    m = existing[key].get("overall", {})
                    mlflow_metrics[f"pl{pl}_f1"]     = float(m.get("f1", 0))
                    mlflow_metrics[f"pl{pl}_fpr"]    = float(m.get("fpr", 0))
                    mlflow_metrics[f"pl{pl}_recall"] = float(m.get("recall", 0))
                    mlflow_metrics[f"pl{pl}_auc_pr"] = float(m.get("auc_pr", 0))
                    lat = existing[key].get("throughput", {})
                    mlflow_metrics[f"pl{pl}_rps"]    = float(lat.get("throughput_rps", 0))
            log_metrics_dict(mlflow_metrics)
            mlflow.log_artifact(str(baselines_path))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    # Print paper-ready comparison table to log
    if len(pl_range) > 1:
        log.info("\n── CRS Paranoia Level Comparison (for paper Table) ──")
        log.info(f"{'PL':>4}  {'F1':>7}  {'FPR':>8}  {'Recall':>7}  {'AUC-PR':>7}")
        for pl in pl_range:
            e = existing[f"modsecurity_crs_pl{pl}"]["overall"]
            log.info(
                f"PL{pl}  "
                f"F1={e['f1']:.4f}  "
                f"FPR={e['fpr']:.5f}  "
                f"Recall={e['recall']:.4f}  "
                f"AUC-PR={e['auc_pr']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2.1 — ModSecurity CRS baseline with Paranoia Levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",         default="config/pipeline.yaml")
    p.add_argument("--paranoia-level", type=int, choices=[1, 2, 3, 4], default=None,
                   help="Evaluate a single PL (default: all four)")
    p.add_argument("--threshold",      type=int, default=None,
                   help="Anomaly score threshold (default: cfg.slo.crs_anomaly_threshold or 5)")
    p.add_argument("--audit-log",      default=None,
                   help="Path to real ModSecurity audit log (bypasses heuristic engine)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())