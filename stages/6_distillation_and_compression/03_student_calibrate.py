"""
stages/6_distillation_and_compression/03_student_calibrate.py
--------------------------------------------------------------
Stage 6.3 — Calibrate classification thresholds for the distilled student.

WHY THIS IS REQUIRED
--------------------
Distillation shifts the student's logit distribution relative to the teacher.
Deploying the student with the teacher's calibrated threshold will inflate the
False Positive Rate (FPR). This script re-runs the threshold search on the
validation split specifically for the student model.

Replaces: 03_post_training_quant.py (which only benchmarked PTQ on the teacher
and provided no actionable threshold for the student).

Method:
  - Collect student logit scores on the validation set.
  - Sweep decision thresholds and compute precision, recall, F1, FPR.
  - Select the threshold that maximises F1 subject to FPR ≤ cfg.slo.max_fpr.
  - Save calibrated threshold to reports/metrics/student_threshold.json.

Run:
    python stages/6_distillation_and_compression/03_student_calibrate.py \
        --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _compute_metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr       = fp / max(fp + tn, 1)
    return {"threshold": threshold, "f1": f1, "precision": precision,
            "recall": recall, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 200,
) -> list[dict]:
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    return [_compute_metrics_at_threshold(scores, labels, float(t)) for t in thresholds]


def _select_threshold(results: list[dict], max_fpr: float) -> dict:
    """Return the result with highest F1 where FPR ≤ max_fpr."""
    feasible = [r for r in results if r["fpr"] <= max_fpr]
    if not feasible:
        log.warning(
            f"No threshold satisfies FPR ≤ {max_fpr:.4f}. "
            "Falling back to the threshold with lowest FPR overall."
        )
        return min(results, key=lambda r: r["fpr"])
    return max(feasible, key=lambda r: r["f1"])


# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_scores(
    model: StudentClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax_attack_scores, labels) arrays from the full dataloader."""
    model.eval()
    all_scores, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(input_ids, attention_mask)

        logits = out["logits"].float().cpu()
        probs  = torch.softmax(logits, dim=-1)
        # Class index 1 is "attack" by convention
        scores = probs[:, 1].numpy()

        all_scores.append(scores)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    student_cfg = cfg.model.student

    require_inputs({
        f"{student_cfg.output_dir}/best_student.pt": "run 02_distill_train.py",
        "data/splits/val.parquet": "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "student_threshold.json",
        args.force, "Stage 6.3 student calibration"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer  = StepTimer()

    # ── Load student ──────────────────────────────
    checkpoint = Path(student_cfg.output_dir) / "best_student.pt"
    if not checkpoint.exists():
        log.error(
            f"Student checkpoint not found: {checkpoint}. "
            "Run 02_distill_train.py first."
        )
        return

    log.info(f"Loading student from {checkpoint}...")
    student = StudentClassifier.load(checkpoint, student_cfg, map_location=str(device))
    student.to(device).eval()
    log.info(f"Student: {student.count_parameters():,} parameters")

    # ── Data ──────────────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )
    collator = WafCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.tokenizer.seq_len,
    )
    val_loader = DataLoader(
        WafDataset(
            get_split_path(cfg.paths.data_splits, "val"),
            tokenizer._tok,
            cfg.tokenizer.seq_len,
        ),
        batch_size=256,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # ── Score collection ──────────────────────────
    log.info("Collecting student scores on validation set...")
    with timer.step("collect_scores"):
        scores, labels = _collect_scores(student, val_loader, device)
    log.info(f"Collected {len(scores):,} samples — "
             f"{labels.sum():,} attacks / {(1-labels).sum():,} benign")

    # ── Threshold sweep ───────────────────────────
    max_fpr = getattr(cfg.slo, "max_fpr", 0.005)
    with timer.step("threshold_sweep"):
        sweep = _sweep_thresholds(scores, labels, n_steps=400)
        best  = _select_threshold(sweep, max_fpr=max_fpr)

    log.info("=" * 60)
    log.info("Student Threshold Calibration Results")
    log.info("=" * 60)
    log.info(f"  Selected threshold : {best['threshold']:.4f}")
    log.info(f"  F1                 : {best['f1']:.4f}")
    log.info(f"  Precision          : {best['precision']:.4f}")
    log.info(f"  Recall             : {best['recall']:.4f}")
    log.info(f"  FPR                : {best['fpr']:.4f}  (SLO: ≤{max_fpr})")
    log.info(f"  TP/FP/FN/TN        : {best['tp']}/{best['fp']}/{best['fn']}/{best['tn']}")
    log.info("=" * 60)

    if best["fpr"] > max_fpr:
        log.warning(
            f"CALIBRATION WARNING: FPR {best['fpr']:.4f} exceeds SLO {max_fpr}. "
            "Consider retraining with higher alpha_hard or more benign samples."
        )

    # ── Save ──────────────────────────────────────
    out_dir = Path(cfg.paths.reports) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "student_threshold.json"
    out_path.write_text(json.dumps({
        "calibrated_threshold": best["threshold"],
        "metrics_at_threshold": best,
        "slo_max_fpr":          max_fpr,
        "slo_satisfied":        best["fpr"] <= max_fpr,
        "n_val_samples":        int(len(scores)),
        "full_sweep":           sweep,
        "timings_s":            timer.timings,
    }, indent=2))

    log.info(f"Student threshold saved to {out_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="03_student_calibrate"):
            mlflow.log_params({
                "max_fpr":          max_fpr,
                "n_sweep_steps":    400,
                "n_val_samples":    int(len(scores)),
            })
            log_metrics_dict({
                "calibrated_threshold": float(best["threshold"]),
                "f1_at_threshold":      float(best["f1"]),
                "precision":            float(best["precision"]),
                "recall":               float(best["recall"]),
                "fpr":                  float(best["fpr"]),
                "slo_satisfied":        float(best["fpr"] <= max_fpr),
            })
            mlflow.log_artifact(str(out_path))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate student classification threshold.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())