"""
stages/7_evaluation/01_detection_metrics.py
-------------------------------------
Stage 7.1 — Run full detection efficacy evaluation for ALL trained models
on the test split.  Produces the master comparison table (Table 1).

Evaluates:
  - Track B 99M    (encoder from scratch)
  - Student        (distilled INT8)
  - Baseline XGBoost (from saved report)
  - Baseline ModSecurity CRS (from saved report)

Metrics per model AND per attack class:
  F1, Precision, Recall, FPR, FNR, AUC-ROC, AUC-PR

Enhancement (critique §4 + §5):
  - FP diagnostic categorisation: "Safe-but-Malformed" vs "High-Entropy"
    detected via Shannon entropy of the raw byte sequence.
  - K-Means clustering on FP embeddings (CLS token) with PCA projection
    to identify systematic misclassification patterns for targeted
    augmentation in Stage 2.

Run:
    python stages/7_evaluation/01_detection_metrics.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.eval.metrics import compute_metrics, compute_per_class_metrics
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config, ModelArchConfig, StudentModelConfig
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# ── FP categorisation thresholds ─────────────────────────────────────────────
# Shannon entropy (bits/byte) above this → "High-Entropy" (encrypted / serialized)
HIGH_ENTROPY_THRESHOLD = 5.5
FP_CLUSTER_K = 5          # K-Means clusters for FP embedding analysis
FP_PCA_COMPONENTS = 32    # PCA dims before K-Means (speeds up clustering)


# ─────────────────────────────────────────────────────────────────────────────
# Core model loading + inference
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_predict(
    checkpoint: Path,
    arch: ModelArchConfig | StudentModelConfig,
    tokenizer: HttpTokenizer,
    test_loader: DataLoader,
    device: torch.device,
    is_student: bool = False,
) -> dict[str, Any]:
    """Load a model from checkpoint and run inference on test_loader.

    Returns overall + per-class metrics, plus raw arrays needed for FP analysis.
    """
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

    all_preds:    list[torch.Tensor] = []
    all_probs:    list[torch.Tensor] = []
    all_labels:   list[torch.Tensor] = []
    all_classes:  list[str]          = []
    all_cls_embs: list[torch.Tensor] = []   # CLS embeddings for FP clustering
    all_raw_texts: list[str]         = []   # raw strings for entropy categorisation

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with autocast:
                preds, probs = model.predict(ids, mask)
                # Extract CLS embedding (position 0 of encoder output)
                # Works for WafClassifier; gracefully skips if attribute absent
                if hasattr(model, "encoder"):
                    enc_out = model.encoder(ids, mask)   # (B, T, D)
                    cls_emb = enc_out[:, 0, :].float()   # (B, D)
                    all_cls_embs.append(cls_emb.cpu())

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"])
            if "attack_class" in batch:
                all_classes.extend(batch["attack_class"])
            if "raw" in batch:
                all_raw_texts.extend(batch["raw"])

    preds_t  = torch.cat(all_preds)
    probs_t  = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)

    overall = compute_metrics(preds_t, probs_t, labels_t)
    per_cls = (
        compute_per_class_metrics(preds_t, probs_t, labels_t, all_classes)
        if all_classes else {}
    )

    result: dict[str, Any] = {"overall": overall, "per_class": per_cls}

    # ── FP analysis (critique §4 + §5) ──────────────────────────────────────
    fp_mask = (labels_t == 0) & (preds_t == 1)
    if fp_mask.sum() > 0 and all_cls_embs and all_raw_texts:
        result["fp_analysis"] = _analyse_false_positives(
            fp_mask=fp_mask,
            cls_embs=torch.cat(all_cls_embs),
            raw_texts=all_raw_texts,
            probs=probs_t,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FP categorisation and clustering
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    """Compute per-byte Shannon entropy (bits) of the UTF-8 encoded string."""
    data = text.encode("utf-8", errors="replace")
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(float)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _categorise_fp(text: str) -> str:
    """Assign a FP to one of two diagnostic buckets.

    - "high_entropy"      → entropy > threshold (encrypted blobs, serialized objects)
    - "safe_but_malformed" → legitimate but syntactically unusual requests
    """
    if _shannon_entropy(text) >= HIGH_ENTROPY_THRESHOLD:
        return "high_entropy"
    return "safe_but_malformed"


def _analyse_false_positives(
    fp_mask: torch.Tensor,
    cls_embs: torch.Tensor,
    raw_texts: list[str],
    probs: torch.Tensor,
) -> dict[str, Any]:
    """
    Critique §4 — Static FP Categorisation:
        Classify each FP as 'high_entropy' or 'safe_but_malformed'.

    Critique §5 — FP Clustering:
        Run PCA → K-Means on CLS embeddings of FP samples.
        Returns cluster centroids, size, and representative texts per cluster.
    """
    fp_indices  = fp_mask.nonzero(as_tuple=True)[0].tolist()
    fp_texts    = [raw_texts[i] for i in fp_indices if i < len(raw_texts)]
    fp_embs     = cls_embs[fp_mask]          # (N_fp, D)
    fp_probs    = probs[fp_mask].tolist()

    # ── Categorical bucketing ────────────────────────────────────────────────
    categories: dict[str, int] = {"high_entropy": 0, "safe_but_malformed": 0}
    fp_categories: list[str] = []
    for t in fp_texts:
        cat = _categorise_fp(t)
        categories[cat] += 1
        fp_categories.append(cat)

    analysis: dict[str, Any] = {
        "total_fp": int(fp_mask.sum()),
        "category_counts": categories,
        "high_entropy_threshold_bits": HIGH_ENTROPY_THRESHOLD,
    }

    # ── K-Means clustering on embeddings ────────────────────────────────────
    n_fp = fp_embs.shape[0]
    if n_fp >= FP_CLUSTER_K:
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans

            emb_np = fp_embs.numpy()
            n_components = min(FP_PCA_COMPONENTS, n_fp, emb_np.shape[1])
            pca   = PCA(n_components=n_components, random_state=42)
            emb_r = pca.fit_transform(emb_np)           # (N_fp, n_components)

            k     = min(FP_CLUSTER_K, n_fp)
            km    = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels_km = km.fit_predict(emb_r)

            clusters: list[dict] = []
            for cid in range(k):
                idx_in_cluster = [i for i, l in enumerate(labels_km) if l == cid]
                cluster_texts  = [fp_texts[i] for i in idx_in_cluster if i < len(fp_texts)]
                cluster_cats   = [fp_categories[i] for i in idx_in_cluster if i < len(fp_categories)]
                cat_counts     = {
                    "high_entropy":       cluster_cats.count("high_entropy"),
                    "safe_but_malformed": cluster_cats.count("safe_but_malformed"),
                }
                # Most confident FP in this cluster (highest malicious prob)
                top_prob_idx  = max(idx_in_cluster, key=lambda i: fp_probs[i])
                representative = fp_texts[top_prob_idx][:200] if top_prob_idx < len(fp_texts) else ""

                clusters.append({
                    "cluster_id":        cid,
                    "size":              len(idx_in_cluster),
                    "category_counts":   cat_counts,
                    "representative_fp": representative,
                    "avg_prob_malicious": round(
                        float(np.mean([fp_probs[i] for i in idx_in_cluster])), 4
                    ),
                })

            analysis["pca_variance_explained"] = round(
                float(pca.explained_variance_ratio_.sum()), 4
            )
            analysis["kmeans_clusters"] = clusters
            log.info(f"  FP clustering: {k} clusters from {n_fp} FP samples "
                     f"(PCA variance={analysis['pca_variance_explained']:.3f})")

        except ImportError:
            log.warning("scikit-learn not available — skipping FP clustering")
        except Exception as exc:
            log.warning(f"FP clustering failed: {exc}")
    else:
        log.info(f"  Too few FP samples ({n_fp}) for K-Means — skipping clustering")

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir  = cfg.paths.data_splits
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    collator = WafCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.tokenizer.seq_len,
        include_attack_class=True,
        include_raw=True,       # needed for FP entropy categorisation
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

    # ── Track B 99M ───────────────────────────────────────────────────────────
    log.info("Evaluating Track B 99M...")
    results["track_b_99m"] = _load_model_and_predict(
        Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt",
        cfg.model.track_b_99m,
        tokenizer, test_loader, device,
    )
    _log_result("track_b_99m", results["track_b_99m"])

    # ── Student (distilled) ───────────────────────────────────────────────────
    log.info("Evaluating distilled student...")
    results["student"] = _load_model_and_predict(
        Path(cfg.model.student.output_dir) / "best_student.pt",
        cfg.model.student,
        tokenizer, test_loader, device,
        is_student=True,
    )
    _log_result("student", results["student"])

    # ── Baselines ─────────────────────────────────────────────────────────────
    baseline_path = reports_dir / "baselines.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
        for name, data in baselines.items():
            results[name] = data
            log.info(f"Loaded baseline: {name}")
    else:
        log.warning("baselines.json not found — run Stage 0 first")

    # ── Comparison table ──────────────────────────────────────────────────────
    log.info("\n=== COMPARISON TABLE ===")
    _print_comparison_table(results)

    # ── FP diagnostic summary ─────────────────────────────────────────────────
    log.info("\n=== FP DIAGNOSTIC SUMMARY ===")
    for name, data in results.items():
        fp = data.get("fp_analysis")
        if fp:
            log.info(
                f"  {name:25s}: total_fp={fp['total_fp']}  "
                f"high_entropy={fp['category_counts'].get('high_entropy', 0)}  "
                f"safe_malformed={fp['category_counts'].get('safe_but_malformed', 0)}  "
                f"clusters={len(fp.get('kmeans_clusters', []))}"
            )

    (reports_dir / "detection_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    log.info(f"\nFull results saved to {reports_dir / 'detection_results.json'}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_detection_metrics"):
            mlflow.log_params({
                "high_entropy_threshold": HIGH_ENTROPY_THRESHOLD,
                "fp_cluster_k":           FP_CLUSTER_K,
                "fp_pca_components":      FP_PCA_COMPONENTS,
                "n_models_evaluated":     len(results),
            })
            metrics: dict[str, float] = {}
            for model_name, model_data in results.items():
                m = model_data.get("overall", model_data.get("val_metrics", {}))
                if not m:
                    continue
                prefix = model_name
                for key in ("f1", "precision", "recall", "fpr", "auc_pr", "auc_roc"):
                    val = m.get(key)
                    if isinstance(val, (int, float)):
                        metrics[f"{prefix}_{key}"] = float(val)
                fp = model_data.get("fp_analysis", {})
                if fp:
                    metrics[f"{prefix}_total_fp"] = float(fp.get("total_fp", 0))
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(reports_dir / "detection_results.json"))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


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
            f"{m.get('f1', 0):>7.4f}  "
            f"{m.get('precision', 0):>7.4f}  "
            f"{m.get('recall', 0):>7.4f}  "
            f"{m.get('fpr', 0):>9.5f}  "
            f"{m.get('auc_pr', 0):>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())