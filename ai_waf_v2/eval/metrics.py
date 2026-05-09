"""
ai_waf_v2.eval.metrics
------------------
Detection efficacy metrics for binary WAF classification.

Primary metrics (all reported per model AND per attack class):
    F1, Precision, Recall, FPR, FNR, AUC-ROC, AUC-PR

Usage
-----
    from ai_waf_v2.eval.metrics import compute_metrics, compute_per_class_metrics

    metrics = compute_metrics(preds, probs, labels)
    # {'f1': 0.97, 'precision': 0.98, 'recall': 0.96, ...}

    per_class = compute_per_class_metrics(preds, probs, labels, attack_classes)
    # {'sqli': {'f1': 0.99, ...}, 'xss': {'f1': 0.95, ...}, ...}
"""

from __future__ import annotations

from collections import defaultdict

import torch


def compute_metrics(
    preds:     torch.Tensor,   # (N,) long — predicted class
    probs:     torch.Tensor,   # (N,) float — P(malicious)
    labels:    torch.Tensor,   # (N,) long — ground truth
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the full suite of binary classification metrics.

    Parameters
    ----------
    preds     : predicted class indices (0 or 1)
    probs     : predicted probability for the positive (malicious) class
    labels    : ground-truth class indices (0 or 1)
    threshold : classification threshold (used only to re-derive preds
                when preds are not already thresholded)

    Returns
    -------
    dict with keys:
        f1, precision, recall, fpr, fnr, accuracy,
        auc_roc, auc_pr, avg_precision
    """
    preds  = preds.cpu().float()
    probs  = probs.cpu().float()
    labels = labels.cpu().float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)   # TPR / sensitivity
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn + 1e-9)   # false positive rate
    fnr       = fn / (fn + tp + 1e-9)   # false negative rate
    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-9)

    auc_roc     = _auc_roc(probs, labels)
    auc_pr      = _auc_pr(probs, labels)
    avg_prec    = _average_precision(probs, labels)

    return {
        "f1":            round(f1,        6),
        "precision":     round(precision, 6),
        "recall":        round(recall,    6),
        "fpr":           round(fpr,       6),
        "fnr":           round(fnr,       6),
        "accuracy":      round(accuracy,  6),
        "auc_roc":       round(auc_roc,   6),
        "auc_pr":        round(auc_pr,    6),
        "avg_precision": round(avg_prec,  6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def compute_per_class_metrics(
    preds:          torch.Tensor,   # (N,)
    probs:          torch.Tensor,   # (N,)
    labels:         torch.Tensor,   # (N,)
    attack_classes: list[str],      # length N — "benign", "sqli", ...
) -> dict[str, dict[str, float]]:
    """
    Compute metrics separately for each attack class.

    For attack classes (non-benign), only their samples + all benign
    samples are included — this mirrors the realistic detection scenario
    where each attack type is evaluated against the full benign pool.

    Returns
    -------
    dict mapping attack_class → metrics dict (same keys as compute_metrics)
    """
    preds  = preds.cpu()
    probs  = probs.cpu()
    labels = labels.cpu()

    # Group indices by attack class
    class_indices: dict[str, list[int]] = defaultdict(list)
    benign_indices: list[int] = []

    for i, cls in enumerate(attack_classes):
        class_indices[cls].append(i)
        if labels[i].item() == 0:
            benign_indices.append(i)

    benign_tensor = torch.tensor(benign_indices, dtype=torch.long)
    results: dict[str, dict[str, float]] = {}

    for cls, indices in class_indices.items():
        if cls == "benign":
            idx = benign_tensor
        else:
            attack_tensor = torch.tensor(indices, dtype=torch.long)
            idx = torch.cat([benign_tensor, attack_tensor])

        if len(idx) == 0:
            continue

        results[cls] = compute_metrics(
            preds[idx], probs[idx], labels[idx]
        )

    return results


def compute_threshold_sweep(
    probs:  torch.Tensor,   # (N,) — P(malicious)
    labels: torch.Tensor,   # (N,)
    n_thresholds: int = 100,
) -> dict[str, list]:
    """
    Sweep classification thresholds and return precision-recall curve data.
    Useful for selecting an operating threshold given a target FPR.

    Returns
    -------
    dict with keys: thresholds, precision, recall, fpr, f1
    """
    thresholds = torch.linspace(0.0, 1.0, n_thresholds)
    out: dict[str, list] = {
        "thresholds": [], "precision": [], "recall": [], "fpr": [], "f1": []
    }

    for t in thresholds:
        preds = (probs >= t).long()
        m = compute_metrics(preds, probs, labels, threshold=t.item())
        out["thresholds"].append(round(t.item(), 4))
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["fpr"].append(m["fpr"])
        out["f1"].append(m["f1"])

    return out


def find_threshold_at_fpr(
    probs:      torch.Tensor,
    labels:     torch.Tensor,
    target_fpr: float = 0.001,
) -> float:
    """
    Find the highest classification threshold that keeps FPR ≤ target_fpr.
    Returns the threshold value (float between 0 and 1).
    """
    sweep = compute_threshold_sweep(probs, labels, n_thresholds=1000)
    best_threshold = 0.5

    for t, fpr in zip(sweep["thresholds"], sweep["fpr"]):
        if fpr <= target_fpr:
            best_threshold = t   # take highest t with acceptable FPR

    return best_threshold


# ─────────────────────────────────────────────────────────
# AUC helpers (pure PyTorch — no sklearn dependency)
# ─────────────────────────────────────────────────────────

def _auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC-ROC using the trapezoidal rule."""
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]

    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0

    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return float(auc)


def _auc_pr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC-PR (area under precision-recall curve)."""
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]

    n_pos = labels.sum().item()
    if n_pos == 0:
        return float("nan")

    precision_list, recall_list = [1.0], [0.0]
    tp = fp = 0

    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        precision_list.append(tp / (tp + fp))
        recall_list.append(tp / n_pos)

    auc = 0.0
    for i in range(1, len(recall_list)):
        auc += (recall_list[i] - recall_list[i - 1]) * (
            precision_list[i] + precision_list[i - 1]
        ) / 2
    return float(auc)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute average precision (AP) — area under PR curve via summation."""
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx].float()

    n_pos = labels.sum().item()
    if n_pos == 0:
        return float("nan")

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    precision = tp / (tp + fp)
    ap = (precision * sorted_labels).sum().item() / n_pos
    return float(ap)