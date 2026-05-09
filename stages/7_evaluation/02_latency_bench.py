"""
stages/7_evaluation/02_latency_bench.py
---------------------------------
Stage 6.2-6.3 — Latency and throughput benchmark for all models
on GPU and CPU at batch sizes [1, 8, 32, 64, 256].

Compares: teacher 99M | student INT8 | ONNX student | baselines
Checks results against SLOs defined in config/pipeline.yaml.

Run:
    python stages/7_evaluation/02_latency_bench.py --config config/pipeline.yaml \
        --batch-sizes 1 8 32 64 256 --devices gpu cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai_waf_v2.eval.latency import LatencyBenchmark, OnnxLatencyBenchmark
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    batch_sizes = args.batch_sizes or cfg.evaluation.batch_sizes
    devices     = args.devices    or cfg.evaluation.devices
    report_dir  = Path(cfg.paths.reports) / "latency"
    report_dir.mkdir(parents=True, exist_ok=True)

    seq_len    = cfg.tokenizer.seq_len
    vocab_size = cfg.model.track_b_99m.vocab_size

    all_results: dict[str, dict] = {}

    for device_str in devices:
        if device_str == "gpu" and not torch.cuda.is_available():
            log.warning("GPU requested but CUDA not available — skipping")
            continue

        device_key = "cuda" if device_str == "gpu" else "cpu"
        device     = torch.device(device_key)

        # ── Track B 99M teacher ───────────────────
        teacher_ckpt = Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt"
        if teacher_ckpt.exists():
            log.info(f"Benchmarking teacher 99M on {device_key}...")
            from ai_waf_v2.models.head import WafClassifier
            teacher = WafClassifier.load(teacher_ckpt, cfg.model.track_b_99m, map_location=device_key)
            teacher.to(device).eval()

            bench   = LatencyBenchmark(
                model=teacher, device=device_key,
                seq_len=seq_len, vocab_size=vocab_size,
                n_warmup=cfg.evaluation.n_latency_warmup,
                n_runs=cfg.evaluation.n_latency_runs,
                use_bf16=(device_key == "cuda"),
            )
            results = bench.run(batch_sizes)
            slo_ok  = bench.check_slo(
                results,
                inline_p99_ms=cfg.slo.latency_inline_p99_ms,
                throughput_min=cfg.slo.throughput_min_rps,
            )
            key = f"teacher_99m_{device_key}"
            all_results[key] = {"results": results, "slo": slo_ok}
            bench.print_table(results)
            bench.save_json(results, report_dir / f"{key}.json")
            log.info(f"  SLO: {slo_ok}")
            del teacher

        # ── Student ───────────────────────────────
        student_ckpt = Path(cfg.model.student.output_dir) / "best_student.pt"
        if student_ckpt.exists():
            log.info(f"Benchmarking student on {device_key}...")
            from ai_waf_v2.models.student import StudentClassifier
            student = StudentClassifier.load(student_ckpt, cfg.model.student, map_location=device_key)
            student.to(device).eval()

            bench   = LatencyBenchmark(
                model=student, device=device_key,
                seq_len=seq_len, vocab_size=vocab_size,
                n_warmup=cfg.evaluation.n_latency_warmup,
                n_runs=cfg.evaluation.n_latency_runs,
                use_bf16=(device_key == "cuda"),
            )
            results = bench.run(batch_sizes)
            slo_ok  = bench.check_slo(results,
                inline_p99_ms=cfg.slo.latency_inline_p99_ms,
                throughput_min=cfg.slo.throughput_min_rps,
            )
            key = f"student_{device_key}"
            all_results[key] = {"results": results, "slo": slo_ok}
            bench.print_table(results)
            bench.save_json(results, report_dir / f"{key}.json")
            log.info(f"  SLO: {slo_ok}")
            del student

        # ── ONNX student ──────────────────────────
        onnx_path = Path(cfg.model.student.output_dir) / "student.onnx"
        if onnx_path.exists() and device_key == "cuda":
            log.info("Benchmarking ONNX Runtime student (GPU)...")
            try:
                ort_bench = OnnxLatencyBenchmark(
                    onnx_path=onnx_path,
                    device=device_key,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    n_warmup=cfg.evaluation.n_latency_warmup,
                    n_runs=cfg.evaluation.n_latency_runs,
                )
                ort_results = ort_bench.run(batch_sizes)
                key = f"student_onnx_{device_key}"
                all_results[key] = {"results": ort_results}
                ort_bench.print_table(ort_results)
                ort_bench.save_json(ort_results, report_dir / f"{key}.json")
            except Exception as e:
                log.warning(f"ONNX benchmark failed: {e}")

    # ── Memory footprint ─────────────────────────
    log.info("\n=== MEMORY FOOTPRINT ===")
    footprints: dict[str, dict] = {}
    for ckpt, label in [
        (Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt", "teacher_99m"),
        (Path(cfg.model.student.output_dir) / "best_student.pt", "student"),
    ]:
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1024 / 1024
            footprints[label] = {"checkpoint_mb": round(size_mb, 1)}
            log.info(f"  {label}: {size_mb:.1f} MB on disk")

    # ── Consolidated report ───────────────────────
    report = {
        "latency_by_model": all_results,
        "memory_footprint":  footprints,
        "slo_targets": {
            "inline_p99_ms":  cfg.slo.latency_inline_p99_ms,
            "throughput_rps": cfg.slo.throughput_min_rps,
        },
    }
    (report_dir / "latency_summary.json").write_text(json.dumps(report, indent=2))
    log.info(f"\nLatency report saved to {report_dir / 'latency_summary.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="config/pipeline.yaml")
    p.add_argument("--batch-sizes",  nargs="+", type=int, default=None)
    p.add_argument("--devices",      nargs="+", default=None, choices=["gpu", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())