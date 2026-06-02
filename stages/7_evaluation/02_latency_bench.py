"""
stages/7_evaluation/02_latency_bench.py
---------------------------------
Stage 7.2 — Latency and throughput benchmark for all models
on GPU and CPU at batch sizes [1, 8, 32, 64, 256].

Compares: teacher 99M | student INT8 | ONNX student | baselines
Checks results against SLOs defined in config/pipeline.yaml.

Enhancements (critique §3):
  - p99.9 tail latency measurement (critical for inline appliance QoS)
  - Per-run jitter (standard deviation) across warmup-excluded iterations
  - PCIe host→device transfer overhead measured separately per batch size
  - All stats exported to latency_summary.json for the decision matrix

Run:
    python stages/7_evaluation/02_latency_bench.py --config config/pipeline.yaml \
        --batch-sizes 1 8 32 64 256 --devices gpu cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai_waf_v2.eval.latency import LatencyBenchmark, OnnxLatencyBenchmark
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

# Number of individual timing samples collected per (model, batch_size) pair.
# Higher → tighter percentile estimates; lower → faster sweep.
N_TIMING_SAMPLES = 500


# ─────────────────────────────────────────────────────────────────────────────
# Extended benchmarking helpers
# ─────────────────────────────────────────────────────────────────────────────

def _measure_pcie_overhead_ms(
    seq_len: int,
    vocab_size: int,
    batch_sizes: list[int],
    n_runs: int = 200,
) -> dict[int, float]:
    """
    Critique §3 — PCIe transfer overhead.

    Measure the host→device transfer time for synthetic input_ids tensors
    at each batch size.  This isolates the PCIe bottleneck from GPU compute
    and is critical for understanding the end-to-end inline latency budget.

    Returns a dict mapping batch_size → median transfer time (ms).
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.device("cuda")
    results: dict[int, float] = {}

    for bs in batch_sizes:
        tensor_cpu = torch.randint(0, vocab_size, (bs, seq_len), dtype=torch.long,
                                   pin_memory=True)
        # Warmup
        for _ in range(20):
            _ = tensor_cpu.to(device, non_blocking=False)
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = tensor_cpu.to(device, non_blocking=False)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        results[bs] = round(float(np.median(times_ms)), 3)
        log.info(f"  PCIe overhead bs={bs:4d}: median={results[bs]:.3f} ms")

    return results


def _extended_latency_stats(
    model: torch.nn.Module,
    device_str: str,
    seq_len: int,
    vocab_size: int,
    batch_sizes: list[int],
    n_warmup: int,
    n_runs: int,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    """
    Critique §3 — Extended latency with p99.9 and jitter.

    For each batch size, collect N_TIMING_SAMPLES end-to-end inference times
    (after warmup), then compute:
      - p50, p99, p99.9 latency
      - standard deviation (jitter)
      - throughput (RPS) based on p50
    """
    device = torch.device(device_str)
    rows: list[dict[str, Any]] = []

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_bf16 and device_str == "cuda")
        else torch.amp.autocast("cuda", enabled=False)
    )

    for bs in batch_sizes:
        ids  = torch.randint(0, vocab_size, (bs, seq_len), dtype=torch.long).to(device)
        mask = torch.ones(bs, seq_len, dtype=torch.long).to(device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad(), autocast_ctx:
                model(ids, mask)
        if device_str == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        times_ms: list[float] = []
        for _ in range(max(n_runs, N_TIMING_SAMPLES)):
            if device_str == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                with torch.no_grad(), autocast_ctx:
                    model(ids, mask)
                end_ev.record()
                torch.cuda.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))
            else:
                t0 = time.perf_counter()
                with torch.no_grad(), autocast_ctx:
                    model(ids, mask)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms)
        p50       = float(np.percentile(arr, 50))
        p99       = float(np.percentile(arr, 99))
        p99_9     = float(np.percentile(arr, 99.9))
        jitter    = float(np.std(arr))
        throughput = round(bs / (p50 / 1000.0), 1)

        row = {
            "batch_size":      bs,
            "p50_ms":          round(p50, 3),
            "p99_ms":          round(p99, 3),
            "p99_9_ms":        round(p99_9, 3),
            "jitter_std_ms":   round(jitter, 3),
            "throughput_rps":  throughput,
            "n_samples":       len(arr),
        }
        rows.append(row)
        log.info(
            f"  bs={bs:4d}: p50={p50:7.2f} ms  p99={p99:7.2f} ms  "
            f"p99.9={p99_9:7.2f} ms  jitter={jitter:6.2f} ms  "
            f"RPS={throughput:,.0f}"
        )

    return rows


def _check_slo(
    stats_rows: list[dict[str, Any]],
    inline_p99_ms: float,
    throughput_min: float,
) -> dict[str, Any]:
    """Return SLO pass/fail for p99, p99.9, and throughput at batch_size=1."""
    bs1 = next((r for r in stats_rows if r["batch_size"] == 1), None)
    if bs1 is None:
        return {"error": "no batch_size=1 result"}
    return {
        "p99_ok":    bs1["p99_ms"] <= inline_p99_ms,
        "p99_9_ok":  bs1["p99_9_ms"] <= inline_p99_ms * 1.5,   # 1.5× budget for tail
        "rps_ok":    bs1["throughput_rps"] >= throughput_min,
        "p99_ms":    bs1["p99_ms"],
        "p99_9_ms":  bs1["p99_9_ms"],
        "rps":       bs1["throughput_rps"],
        "target_p99_ms":  inline_p99_ms,
        "target_rps":     throughput_min,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    require_inputs({
        f"{cfg.model.track_b_99m.output_dir}/best_99m.pt": "run 00_train_teacher_99m.py",
    })
    if check_output(
        Path(cfg.paths.reports) / "latency" / "latency_summary.json",
        args.force, "Stage 7.2 latency benchmark"
    ):
        return

    batch_sizes = args.batch_sizes or cfg.evaluation.batch_sizes
    devices     = args.devices    or cfg.evaluation.devices
    report_dir  = Path(cfg.paths.reports) / "latency"
    report_dir.mkdir(parents=True, exist_ok=True)

    seq_len    = cfg.tokenizer.seq_len
    vocab_size = cfg.model.track_b_99m.vocab_size

    all_results: dict[str, dict] = {}
    timer = StepTimer()

    # ── PCIe overhead (GPU only, once) ────────────────────────────────────────
    pcie_overhead: dict[str, float] = {}
    if "gpu" in devices and torch.cuda.is_available():
        log.info("\n=== PCIe HOST→DEVICE TRANSFER OVERHEAD ===")
        with timer.step("pcie_overhead"):
            pcie_overhead = {
                str(bs): v
                for bs, v in _measure_pcie_overhead_ms(seq_len, vocab_size, batch_sizes).items()
            }

    for device_str_cfg in devices:
        if device_str_cfg == "gpu" and not torch.cuda.is_available():
            log.warning("GPU requested but CUDA not available — skipping")
            continue

        device_key = "cuda" if device_str_cfg == "gpu" else "cpu"
        device     = torch.device(device_key)
        use_bf16   = (device_key == "cuda")

        # ── Track B 99M teacher ───────────────────────────────────────────────
        teacher_ckpt = Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt"
        if teacher_ckpt.exists():
            log.info(f"\n=== Teacher 99M on {device_key} ===")
            from ai_waf_v2.models.head import WafClassifier
            teacher = WafClassifier.load(teacher_ckpt, cfg.model.track_b_99m,
                                         map_location=device_key)
            teacher.to(device).eval()

            with timer.step(f"benchmark_teacher_{device_key}"):
                stats = _extended_latency_stats(
                    teacher, device_key, seq_len, vocab_size, batch_sizes,
                    n_warmup=cfg.evaluation.n_latency_warmup,
                    n_runs=cfg.evaluation.n_latency_runs,
                    use_bf16=use_bf16,
                )
            slo = _check_slo(stats,
                             cfg.slo.latency_inline_p99_ms,
                             cfg.slo.throughput_min_rps)
            key = f"teacher_99m_{device_key}"
            all_results[key] = {"results": stats, "slo": slo}
            log.info(f"  SLO: {slo}")
            (report_dir / f"{key}.json").write_text(json.dumps({"results": stats, "slo": slo}, indent=2))
            del teacher

        # ── Student ───────────────────────────────────────────────────────────
        student_ckpt = Path(cfg.model.student.output_dir) / "best_student.pt"
        if student_ckpt.exists():
            log.info(f"\n=== Student on {device_key} ===")
            from ai_waf_v2.models.student import StudentClassifier
            student = StudentClassifier.load(student_ckpt, cfg.model.student,
                                             map_location=device_key)
            student.to(device).eval()

            with timer.step(f"benchmark_student_{device_key}"):
                stats = _extended_latency_stats(
                    student, device_key, seq_len, vocab_size, batch_sizes,
                    n_warmup=cfg.evaluation.n_latency_warmup,
                    n_runs=cfg.evaluation.n_latency_runs,
                    use_bf16=use_bf16,
                )
            slo = _check_slo(stats,
                             cfg.slo.latency_inline_p99_ms,
                             cfg.slo.throughput_min_rps)
            key = f"student_{device_key}"
            all_results[key] = {"results": stats, "slo": slo}
            log.info(f"  SLO: {slo}")
            (report_dir / f"{key}.json").write_text(json.dumps({"results": stats, "slo": slo}, indent=2))
            del student

        # ── ONNX student ──────────────────────────────────────────────────────
        onnx_path = Path(cfg.model.student.output_dir) / "student.onnx"
        if onnx_path.exists() and device_key == "cuda":
            log.info("\n=== ONNX Runtime student (GPU) ===")
            try:
                ort_bench   = OnnxLatencyBenchmark(
                    onnx_path=onnx_path, device=device_key,
                    seq_len=seq_len, vocab_size=vocab_size,
                    n_warmup=cfg.evaluation.n_latency_warmup,
                    n_runs=max(cfg.evaluation.n_latency_runs, N_TIMING_SAMPLES),
                )
                with timer.step("benchmark_student_onnx"):
                    ort_results = ort_bench.run(batch_sizes)
                key = f"student_onnx_{device_key}"
                all_results[key] = {"results": ort_results}
                ort_bench.print_table(ort_results)
                (report_dir / f"{key}.json").write_text(
                    json.dumps({"results": ort_results}, indent=2)
                )
            except Exception as e:
                log.warning(f"ONNX benchmark failed: {e}")

    # ── Memory footprint ──────────────────────────────────────────────────────
    log.info("\n=== MEMORY FOOTPRINT ===")
    footprints: dict[str, dict] = {}
    for ckpt, label in [
        (Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt", "teacher_99m"),
        (Path(cfg.model.student.output_dir) / "best_student.pt",  "student"),
    ]:
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1024 / 1024
            footprints[label] = {"checkpoint_mb": round(size_mb, 1)}
            log.info(f"  {label}: {size_mb:.1f} MB on disk")

    # ── Consolidated report ───────────────────────────────────────────────────
    report: dict[str, Any] = {
        "latency_by_model": all_results,
        "memory_footprint":  footprints,
        "pcie_overhead_ms":  pcie_overhead,
        "slo_targets": {
            "inline_p99_ms":  cfg.slo.latency_inline_p99_ms,
            "throughput_rps": cfg.slo.throughput_min_rps,
        },
        "methodology": {
            "n_timing_samples":  N_TIMING_SAMPLES,
            "percentiles_reported": ["p50", "p99", "p99.9"],
            "jitter_metric": "std_dev_ms",
        },
        "timings_s": timer.timings,
    }
    (report_dir / "latency_summary.json").write_text(json.dumps(report, indent=2))
    log.info(f"\nLatency report saved to {report_dir / 'latency_summary.json'}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="02_latency_bench") as _run:
            mlflow.log_params({
                "n_timing_samples":  N_TIMING_SAMPLES,
                "batch_sizes":       str(batch_sizes),
                "devices":           str(devices),
                "inline_p99_slo_ms": cfg.slo.latency_inline_p99_ms,
                "throughput_slo_rps": cfg.slo.throughput_min_rps,
            })
            metrics: dict[str, float] = {}
            for model_key, model_val in all_results.items():
                results_list = model_val.get("results", [])
                bs1 = next((r for r in results_list if r.get("batch_size") == 1), None)
                if bs1:
                    metrics[f"{model_key}_p99_ms"]      = float(bs1["p99_ms"])
                    metrics[f"{model_key}_p99_9_ms"]    = float(bs1.get("p99_9_ms", 0))
                    metrics[f"{model_key}_jitter_ms"]   = float(bs1.get("jitter_std_ms", 0))
                    metrics[f"{model_key}_rps"]         = float(bs1["throughput_rps"])
                slo = model_val.get("slo", {})
                if slo:
                    metrics[f"{model_key}_slo_p99_ok"]  = float(slo.get("p99_ok", False))
                    metrics[f"{model_key}_slo_rps_ok"]  = float(slo.get("rps_ok", False))
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(report_dir / "latency_summary.json"))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config/pipeline.yaml")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    p.add_argument("--devices",     nargs="+", default=None, choices=["gpu", "cpu"])
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())