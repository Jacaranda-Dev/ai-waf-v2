"""
ai_waf_v2/utils/reports.py
--------------------------
Single source of truth for where every report artefact lives under `reports/`.

Reports are organised by pipeline stage (mirroring `stages/`), keeping the
`metrics/` · `latency/` · `figures/` split inside each stage, and numbered by the
substage that owns them:

    reports/<N>_<stage>/<metrics|latency|figures>/<NN>_<name>

Both the writer and every downstream reader resolve paths through `report_path`
(or `report_dir`), so the layout is defined here once and the two can never drift
— important because later stages read earlier stages' reports (the augmentation
governor reads `taxonomy_inventory`, framing reads `traffic_distribution`,
evaluation reads the stage-2 `baselines`, etc.).

To reorganise the tree in future, edit `REPORTS` only.
"""

from __future__ import annotations

from pathlib import Path

# Logical report basename → path relative to `cfg.paths.reports`.
# A report written by several substages (e.g. taxonomy_inventory, split_stats)
# has ONE canonical entry so all writers/readers agree.
REPORTS: dict[str, str] = {
    # ── Stage 1 — acquisition & curation ──
    "download_manifest.json":        "1_data_acquisition_and_curation/metrics/01_download_manifest.json",
    "collection_stats.json":         "1_data_acquisition_and_curation/metrics/01_collection_stats.json",
    "normalization_stats.json":      "1_data_acquisition_and_curation/metrics/01_normalization_stats.json",
    "dedup_stats.json":              "1_data_acquisition_and_curation/metrics/02_dedup_stats.json",
    "corpus_report.json":            "1_data_acquisition_and_curation/metrics/03_corpus_report.json",
    "dataset_analysis.json":         "1_data_acquisition_and_curation/metrics/03_dataset_analysis.json",
    "datasheet.json":                "1_data_acquisition_and_curation/metrics/03_datasheet.json",
    "length_distribution.json":      "1_data_acquisition_and_curation/metrics/03_length_distribution.json",
    "taxonomy_coverage.json":        "1_data_acquisition_and_curation/metrics/03_taxonomy_coverage.json",
    "taxonomy_inventory.json":       "1_data_acquisition_and_curation/metrics/03_taxonomy_inventory.json",

    # ── Stage 2 — baselines ──
    "split_stats.json":              "2_baselines/metrics/00_split_stats.json",
    "slos.json":                     "2_baselines/metrics/01_slos.json",
    "modsecurity_results.json":      "2_baselines/metrics/02_modsecurity_results.json",
    "baselines.json":                "2_baselines/metrics/03_baselines.json",

    # ── Stage 3 — augmentation ──
    "augmentation_synthesis.json":   "3_data_augmentation/metrics/01_augmentation_synthesis.json",
    "traffic_distribution.json":     "3_data_augmentation/metrics/02_traffic_distribution.json",
    "traffic_profiler.json":         "3_data_augmentation/metrics/02_traffic_profiler.json",
    "request_framing.json":          "3_data_augmentation/metrics/03_request_framing.json",
    "quality_gate.json":             "3_data_augmentation/metrics/04_quality_gate.json",
    "augmentation_probe.json":       "3_data_augmentation/metrics/05_augmentation_probe.json",
    "token_leakage.json":            "3_data_augmentation/metrics/08_token_leakage.json",

    # ── Stage 4 — tokenization ──
    "tokenizer_track_a.json":        "4_tokenization/metrics/01_tokenizer_track_a.json",
    "tokenizer_oov_track_a.json":    "4_tokenization/metrics/02_tokenizer_oov_track_a.json",
    "tokenizer_oov_track_b.json":    "4_tokenization/metrics/04_tokenizer_oov_track_b.json",
    "tokenizer_comparison.json":     "4_tokenization/metrics/05_tokenizer_comparison.json",

    # ── Stage 6 — distillation & compression ──
    "student_arch.json":             "6_distillation_and_compression/metrics/01_student_arch.json",
    "student_threshold.json":        "6_distillation_and_compression/metrics/03_student_threshold.json",
    "student_canary.json":           "6_distillation_and_compression/metrics/04_student_canary.json",
    "onnx_bench.json":               "6_distillation_and_compression/latency/05_onnx_bench.json",
    "trt_bench.json":                "6_distillation_and_compression/latency/05_trt_bench.json",
    "ort_detailed_bench.json":       "6_distillation_and_compression/latency/05_ort_detailed_bench.json",

    # ── Stage 7 — evaluation ──
    "detection_results.json":            "7_evaluation/metrics/01_detection_results.json",
    "latency_summary.json":              "7_evaluation/latency/02_latency_summary.json",
    "memory_footprint.json":             "7_evaluation/metrics/03_memory_footprint.json",
    "adversarial_summary.json":          "7_evaluation/metrics/04_adversarial_summary.json",
    "obfuscation_robustness.json":       "7_evaluation/metrics/05_obfuscation_robustness.json",
    "novel_attack_generalization.json":  "7_evaluation/metrics/06_novel_attack_generalization.json",
    "tokenizer_ablation.json":           "7_evaluation/metrics/07_tokenizer_ablation.json",
    "augmentation_ablation.json":        "7_evaluation/metrics/08_augmentation_ablation.json",
    "model_size_scaling.json":           "7_evaluation/metrics/09_model_size_scaling.json",
    "label_smoothing_ablation.json":     "7_evaluation/metrics/10_label_smoothing_ablation.json",
    "attention_visualization.json":      "7_evaluation/metrics/11_attention_visualization.json",
    "shap_analysis.json":                "7_evaluation/metrics/12_shap_analysis.json",
    "error_analysis.json":               "7_evaluation/metrics/13_error_analysis.json",
    "baselines_post_aug.json":           "7_evaluation/metrics/14_baselines_post_aug.json",
    "master_comparison_table.json":      "7_evaluation/metrics/14_master_comparison_table.json",
    "master_comparison_table.csv":       "7_evaluation/metrics/14_master_comparison_table.csv",
    "final_evaluation_report.json":      "7_evaluation/15_final_evaluation_report.json",
    "deployment_recommendation.json":    "7_evaluation/metrics/16_deployment_recommendation.json",
}

# Directory artefacts (e.g. per-head attention heatmaps).
REPORT_DIRS: dict[str, str] = {
    "attention_heatmaps": "7_evaluation/figures/11_attention_heatmaps",
}


def report_path(name: str, reports_root: str | Path, mkdir: bool = True) -> Path:
    """
    Resolve a report basename to its full path under `reports_root`
    (= `cfg.paths.reports`). Creates the parent directory by default.

    Unknown names fall back to `_unfiled/<name>` so a run never crashes over a
    report file — but that path is deliberately conspicuous; register the name in
    REPORTS instead.
    """
    rel = REPORTS.get(name)
    if rel is None:
        rel = f"_unfiled/{name}"
    path = Path(reports_root) / rel
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(name: str, reports_root: str | Path, mkdir: bool = True) -> Path:
    """Resolve a report *directory* (e.g. `attention_heatmaps`)."""
    rel = REPORT_DIRS.get(name, f"_unfiled/{name}")
    path = Path(reports_root) / rel
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path
