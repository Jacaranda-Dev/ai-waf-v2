"""
stages/7_evaluation/01_detection_metrics.py
-------------------------------------
Stage 6.1 — Run full detection efficacy evaluation for ALL trained models
on the test split.  Produces the master comparison table (Table 1).

Evaluates:
  - Track A large  (DeBERTa fine-tuned)
  - Track A small  (BERT-tiny fine-tuned)
  - Track B 99M    (encoder from scratch)
  - Student        (distilled INT8)
  - Baseline XGBoost (from saved report)
  - Baseline ModSecurity CRS (from saved report)

Metrics per model AND per attack class:
  F1, Precision, Recall, FPR, FNR, AUC-ROC, AUC-PR

Run:
    python stages/7_evaluation/01_detection_metrics.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.eval.metrics import compute_metrics, compute_per_class_metrics
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config, ModelArchConfig, StudentModelConfig
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


def _load_model_and_predict(
    checkpoint: Path,
    arch: ModelArchConfig | StudentModelConfig,
    tokenizer: HttpTokenizer,
    test_loader: DataLoader,
    device: torch.device,
    is_student: bool = False,
) -> dict[str, Any]:
    """Load a model from checkpoint and run inference on test_loader."""
    from ai_waf_v2.models.head import WafClassifier
    from ai_waf_v2.models.student import StudentClassifier

    if not checkpoint.exists():
        return {"error": f"checkpoint not found: {checkpoint}"}

    if is_student:
        model = StudentClassifier.load(checkpoint, arch, map_location=str(device))
    else:
        model = WafClassifier.load(checkpoint, arch, map_location=str(device))

    model.to(device).eval()

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.amp.autocast("cuda", enabled=False)
    )

    all_preds:   list[torch.Tensor] = []
    all_probs:   list[torch.Tensor] = []
    all_labels:  list[torch.Tensor] = []
    all_classes: list[str]          = []

    with torch.no_grad():
        for batch in test_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)

            with autocast:
                preds, probs = model.predict(ids, mask)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"])
            if "attack_class" in batch:
                all_classes.extend(batch["attack_class"])

    preds_t  = torch.cat(all_preds)
    probs_t  = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)

    overall = compute_metrics(preds_t, probs_t, labels_t)
    per_cls = (
        compute_per_class_metrics(preds_t, probs_t, labels_t, all_classes)
        if all_classes else {}
    )

    return {"overall": overall, "per_class": per_cls}


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg        = load_config(args.config)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = cfg.paths.data_splits
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    collator = WafCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.tokenizer.seq_len,
        include_attack_class=True,
    )

    test_loader = DataLoader(
        WafDataset(
            get_split_path(splits_dir, "test"),
            tokenizer=tokenizer._tok,
            seq_len=cfg.tokenizer.seq_len,
        ),
        batch_size=256,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    results: dict[str, Any] = {}

    # ── Track B 99M ───────────────────────────────
    log.info("Evaluating Track B 99M...")
    results["track_b_99m"] = _load_model_and_predict(
        Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt",
        cfg.model.track_b_99m,
        tokenizer, test_loader, device,
    )
    _log_result("track_b_99m", results["track_b_99m"])

    # ── Student (distilled) ───────────────────────
    log.info("Evaluating distilled student...")
    results["student"] = _load_model_and_predict(
        Path(cfg.model.student.output_dir) / "best_student.pt",
        cfg.model.student,
        tokenizer, test_loader, device,
        is_student=True,
    )
    _log_result("student", results["student"])

    # ── Load baseline results (from Stage 0) ──────
    baseline_path = reports_dir / "baselines.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
        for name, data in baselines.items():
            results[name] = data
            log.info(f"Loaded baseline: {name}")
    else:
        log.warning("baselines.json not found — run Stage 0 first")

    # ── Build comparison table ─────────────────────
    log.info("\n=== COMPARISON TABLE ===")
    _print_comparison_table(results)

    # Save full results
    (reports_dir / "detection_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    log.info(f"Full results saved to {reports_dir / 'detection_results.json'}")


def _log_result(name: str, result: dict) -> None:
    if "error" in result:
        log.warning(f"  {name}: {result['error']}")
        return
    m = result.get("overall", {})
    log.info(
        f"  {name:20s}: F1={m.get('f1', 0):.4f}  "
        f"FPR={m.get('fpr', 0):.5f}  "
        f"AUC-PR={m.get('auc_pr', 0):.4f}  "
        f"Recall={m.get('recall', 0):.4f}"
    )


def _print_comparison_table(results: dict) -> None:
    header = f"{'Model':25s}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  {'FPR':>9}  {'AUC-PR':>8}"
    log.info(header)
    log.info("-" * len(header))
    for name, data in results.items():
        m = data.get("overall", data.get("val_metrics", {}))
        if not m:
            continue
        log.info(
            f"{name:25s}  "
            f"{m.get('f1',0):>7.4f}  "
            f"{m.get('precision',0):>7.4f}  "
            f"{m.get('recall',0):>7.4f}  "
            f"{m.get('fpr',0):>9.5f}  "
            f"{m.get('auc_pr',0):>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())