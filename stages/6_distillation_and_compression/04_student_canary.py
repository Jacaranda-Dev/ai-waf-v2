"""
stages/6_distillation_and_compression/04_student_canary.py
----------------------------------------------------------
Stage 6.4 — Synthetic canary evaluation on the distilled student model.

WHY THIS IS REQUIRED
--------------------
Distillation can cause the student to forget rare or syntactically unusual
attack patterns that the teacher detected via its larger capacity. Running
a targeted canary evaluation with synthetic payloads exercises the attack
families most likely to be affected by compression (long-range token
dependencies, obfuscated injections, rare encodings).

This script is ported from Stage 5's 05_canary_eval.py and adapted for:
  - Student inference (lower d_model, INT8 weights)
  - Student-specific calibrated threshold (from 03_student_calibrate.py)
  - Regression gate: fails with exit-code 1 if recall on any attack family
    drops below cfg.slo.canary_min_recall so CI/CD can block bad checkpoints.

Run:
    python stages/6_distillation_and_compression/04_student_canary.py \
        --config config/pipeline.yaml \
        [--canary-dir data/canary_payloads]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import torch
import numpy as np

from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# ── Built-in canary payloads ──────────────────────────────────────────────────
# Minimal synthetic attack corpus used when no canary directory is provided.
# Each tuple: (attack_family, raw_http_snippet)

_BUILTIN_CANARIES: list[tuple[str, str]] = [
    # SQL injection — classic and obfuscated
    ("sqli_classic",    "GET /search?q=1' OR '1'='1 HTTP/1.1"),
    ("sqli_union",      "GET /api?id=1 UNION SELECT null,username,password FROM users-- HTTP/1.1"),
    ("sqli_obfuscated", "GET /item?id=1/*!50000OR*/1=1-- HTTP/1.1"),
    # XSS
    ("xss_basic",       "POST /comment HTTP/1.1\r\n\r\n<script>alert(1)</script>"),
    ("xss_encoded",     "GET /search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E HTTP/1.1"),
    ("xss_event",       "POST /profile HTTP/1.1\r\n\r\n<img src=x onerror=alert(1)>"),
    # Path traversal
    ("path_traversal",  "GET /download?file=../../../../etc/passwd HTTP/1.1"),
    ("path_traversal_encoded", "GET /get?path=..%2F..%2F..%2Fetc%2Fshadow HTTP/1.1"),
    # Command injection
    ("cmdi",            "GET /ping?host=127.0.0.1;cat+/etc/passwd HTTP/1.1"),
    ("cmdi_pipe",       "POST /exec HTTP/1.1\r\n\r\ncmd=ls|whoami"),
    # SSRF
    ("ssrf",            "GET /fetch?url=http://169.254.169.254/latest/meta-data/ HTTP/1.1"),
    ("ssrf_encoded",    "GET /proxy?target=http%3A%2F%2F169.254.169.254%2F HTTP/1.1"),
    # Header injection
    ("header_inject",   "GET / HTTP/1.1\r\nX-Forwarded-For: 127.0.0.1\r\nHost: evil.com"),
    # Long obfuscated payload (stress-tests long-range attention)
    ("long_obfuscated", "GET /search?q=" + "%27%20OR%201%3D1%20--" * 12 + " HTTP/1.1"),
]


# ── Tokenize + score helpers ──────────────────────────────────────────────────

def _batch_score(
    model: StudentClassifier,
    tokenizer: HttpTokenizer,
    texts: list[str],
    threshold: float,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (attack_probs, predictions) for a list of raw HTTP snippets."""
    model.eval()
    all_probs, all_preds = [], []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc   = tokenizer.encode_batch(chunk)
        ids   = torch.tensor(enc["input_ids"],      device=device)
        mask  = torch.tensor(enc["attention_mask"],  device=device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out   = model(ids, mask)
            probs = torch.softmax(out["logits"].float(), dim=-1)[:, 1].cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend((probs >= threshold).astype(int).tolist())

    return np.array(all_probs), np.array(all_preds)


# ── Canary loader ─────────────────────────────────────────────────────────────

def _load_canaries(canary_dir: Path | None) -> list[tuple[str, str]]:
    """Load canary payloads from disk or fall back to built-ins."""
    if canary_dir is None or not canary_dir.exists():
        log.info("No canary directory provided — using built-in synthetic payloads.")
        return _BUILTIN_CANARIES

    canaries: list[tuple[str, str]] = []
    for fpath in sorted(canary_dir.glob("*.jsonl")):
        family = fpath.stem
        for line in fpath.read_text().splitlines():
            item = json.loads(line)
            canaries.append((family, item["text"]))
    if not canaries:
        log.warning(f"No .jsonl files found in {canary_dir}. Falling back to built-ins.")
        return _BUILTIN_CANARIES

    log.info(f"Loaded {len(canaries)} canary payloads from {canary_dir}.")
    return canaries


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    student_cfg = cfg.model.student

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Student ───────────────────────────────────
    checkpoint = Path(student_cfg.output_dir) / "best_student.pt"
    if not checkpoint.exists():
        log.error(f"Student checkpoint not found: {checkpoint}.")
        sys.exit(1)

    log.info(f"Loading student from {checkpoint}...")
    student = StudentClassifier.load(checkpoint, student_cfg, map_location=str(device))
    student.to(device).eval()

    # ── Threshold ─────────────────────────────────
    threshold_path = Path(cfg.paths.reports) / "metrics" / "student_threshold.json"
    if threshold_path.exists():
        thresh_data = json.loads(threshold_path.read_text())
        threshold   = thresh_data["calibrated_threshold"]
        log.info(f"Using student-calibrated threshold: {threshold:.4f}")
    else:
        threshold = getattr(cfg.slo, "default_threshold", 0.5)
        log.warning(
            f"student_threshold.json not found. "
            f"Using default threshold={threshold:.4f}. "
            "Run 03_student_calibrate.py first for accurate FPR control."
        )

    # ── Tokenizer ─────────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    # ── Canaries ──────────────────────────────────
    canary_dir = Path(args.canary_dir) if args.canary_dir else None
    canaries   = _load_canaries(canary_dir)

    # Group by family
    families: dict[str, list[str]] = {}
    for family, text in canaries:
        families.setdefault(family, []).append(text)

    # ── Evaluate per family ───────────────────────
    min_recall  = getattr(cfg.slo, "canary_min_recall", 0.90)
    results     = {}
    failed_fams = []

    log.info("=" * 60)
    log.info("Student Canary Evaluation")
    log.info("=" * 60)

    for family, texts in families.items():
        probs, preds = _batch_score(student, tokenizer, texts, threshold, device)
        recall = float(preds.mean())   # all canaries are attacks → recall = detection rate
        mean_p = float(probs.mean())
        min_p  = float(probs.min())

        status = "PASS" if recall >= min_recall else "FAIL"
        if status == "FAIL":
            failed_fams.append(family)

        log.info(
            f"  [{status}] {family:25s}  recall={recall:.3f}  "
            f"mean_prob={mean_p:.3f}  min_prob={min_p:.3f}  n={len(texts)}"
        )
        results[family] = {
            "recall":    recall,
            "mean_prob": mean_p,
            "min_prob":  min_p,
            "n":         len(texts),
            "pass":      status == "PASS",
        }

    overall_recall = np.mean([r["recall"] for r in results.values()])
    log.info("-" * 60)
    log.info(f"  Overall recall      : {overall_recall:.4f}")
    log.info(f"  SLO min recall      : {min_recall}")
    log.info(f"  Failed families     : {failed_fams or 'none'}")
    log.info("=" * 60)

    # ── Save report ───────────────────────────────
    out_dir = Path(cfg.paths.reports) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "student_canary.json"
    out_path.write_text(json.dumps({
        "threshold":        threshold,
        "min_recall_slo":   min_recall,
        "overall_recall":   float(overall_recall),
        "slo_passed":       len(failed_fams) == 0,
        "failed_families":  failed_fams,
        "per_family":       results,
    }, indent=2))
    log.info(f"Canary report saved to {out_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="04_student_canary"):
            mlflow.log_params({
                "threshold":       threshold,
                "min_recall_slo":  min_recall,
                "n_families":      len(results),
                "n_failed_families": len(failed_fams),
                "canary_source":   "builtin" if (args.canary_dir is None) else "disk",
            })
            metrics: dict[str, float] = {
                "overall_recall": float(overall_recall),
                "slo_passed":     float(len(failed_fams) == 0),
            }
            for fam, info in results.items():
                metrics[f"recall_{fam}"] = float(info["recall"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    # ── CI/CD gate ────────────────────────────────
    if failed_fams:
        log.error(
            f"CANARY GATE FAILED: {len(failed_fams)} attack families below "
            f"recall SLO of {min_recall}. Blocking deployment. "
            f"Failed: {failed_fams}"
        )
        sys.exit(1)

    log.info("Canary gate passed — student is safe to export.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic canary evaluation on student model.")
    p.add_argument("--config",      default="config/pipeline.yaml")
    p.add_argument("--canary-dir",  default=None,
                   help="Path to directory of JSONL canary payload files. "
                        "Falls back to built-in synthetic payloads if not provided.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())