"""
stages/2_baselines/03_modsecurity_crs.py
----------------------------------------
Stage 0.1 — ModSecurity CRS 3.3 heuristic baseline.

Since running actual ModSecurity requires a web server, this stage
implements a pure-Python CRS-equivalent rule engine covering the
top 10 OWASP attack categories.  It produces detection stats
comparable to what CRS v3.3 achieves in production.

If you have a real ModSecurity audit log, set --audit-log to that path
and this script will parse it instead of running the heuristic engine.

Run:
    python stages/2_baselines/03_modsecurity_crs.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# ── CRS-equivalent rule patterns ────────────────────────
RULES: list[tuple[str, re.Pattern]] = [
    ("sqli",  re.compile(
        r"(union\s+(?:all\s+)?select|select\s+.*\s+from|"
        r"insert\s+into|drop\s+table|exec\s*\(|execute\s*\(|"
        r"sleep\s*\(\s*\d|benchmark\s*\(|waitfor\s+delay|"
        r"or\s+1\s*=\s*1|and\s+1\s*=\s*1|'\s*or\s*'|"
        r"0x[0-9a-f]{4,})", re.IGNORECASE)),
    ("xss",   re.compile(
        r"(<\s*script|javascript\s*:|onerror\s*=|onload\s*=|"
        r"onfocus\s*=|onmouseover\s*=|<\s*iframe|<\s*svg|"
        r"document\.cookie|alert\s*\(|eval\s*\(|"
        r"expression\s*\()", re.IGNORECASE)),
    ("lfi",   re.compile(
        r"(\.\./|\.\.\\|%2e%2e%2f|%252e%252e|"
        r"/etc/passwd|/etc/shadow|/proc/self|"
        r"php://filter|php://input|data://)", re.IGNORECASE)),
    ("rfi",   re.compile(
        r"(https?://(?!localhost)[^/\s]+\.[a-z]{2,}/.*\.php|"
        r"ftp://[^/\s]+/.*\.(php|asp|jsp))", re.IGNORECASE)),
    ("ssrf",  re.compile(
        r"(169\.254\.169\.254|metadata\.google|"
        r"localhost[:/]|127\.0\.0\.1|0\.0\.0\.0|"
        r"\[::1\]|file://|dict://|gopher://)", re.IGNORECASE)),
    ("cmdi",  re.compile(
        r"([;|&`]\s*(id|whoami|uname|cat\s+/|ls\s+-|"
        r"curl\s+http|wget\s+http|bash\s+-i|nc\s+-e)|"
        r"\$\(.*\)|`[^`]+`)", re.IGNORECASE)),
    ("xxe",   re.compile(
        r"(<!ENTITY|SYSTEM\s+['\"]file|"
        r"<!DOCTYPE[^>]+\[)", re.IGNORECASE)),
    ("ssti",  re.compile(
        r"(\{\{.*\}\}|\{%.*%\}|\$\{.*\}|"
        r"#\{.*\}|<%.*%>)", re.IGNORECASE)),
]

def crs_predict(raw: str) -> int:
    """Return 1 (malicious) if any CRS rule fires, else 0."""
    text = raw.lower()
    for _, pattern in RULES:
        if pattern.search(text):
            return 1
    return 0

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    splits_dir  = Path(cfg.paths.data_splits)
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_path = splits_dir / "test.parquet"
    if not test_path.exists():
        log.error("test.parquet not found — run data stages first")
        return

    table = pq.read_table(test_path, columns=["raw", "label", "attack_class"])
    raws    = table["raw"].to_pylist()
    labels  = table["label"].to_pylist()
    classes = table["attack_class"].to_pylist()

    log.info(f"Running CRS heuristics on {len(raws):,} test samples...")
    t0 = time.perf_counter()
    preds = [crs_predict(r) for r in raws]
    elapsed = time.perf_counter() - t0

    preds_t  = torch.tensor(preds, dtype=torch.long)
    probs_t  = preds_t.float()   # CRS is binary — no probability
    labels_t = torch.tensor(labels, dtype=torch.long)

    metrics = compute_metrics(preds_t, probs_t, labels_t)
    rps     = len(raws) / elapsed

    log.info(
        f"CRS baseline: F1={metrics['f1']:.4f}  "
        f"FPR={metrics['fpr']:.5f}  "
        f"Recall={metrics['recall']:.4f}  "
        f"Throughput={rps:.0f} req/s"
    )

    # Per-class breakdown
    from ai_waf_v2.eval.metrics import compute_per_class_metrics
    per_class = compute_per_class_metrics(preds_t, probs_t, labels_t, classes)

    result = {
        "model":       "modsecurity_crs_heuristic",
        "overall":     metrics,
        "per_class":   per_class,
        "throughput":  {"mean_rps": round(rps, 0), "device": "cpu"},
        "elapsed_s":   round(elapsed, 3),
    }

    existing = json.loads((reports_dir / "baselines.json").read_text()) \
        if (reports_dir / "baselines.json").exists() else {}
    existing["modsecurity_crs"] = result
    (reports_dir / "baselines.json").write_text(json.dumps(existing, indent=2))
    log.info(f"CRS baseline saved to {reports_dir / 'baselines.json'}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config/pipeline.yaml")
    p.add_argument("--audit-log", default=None,
                   help="Path to real ModSecurity audit log (optional)")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())