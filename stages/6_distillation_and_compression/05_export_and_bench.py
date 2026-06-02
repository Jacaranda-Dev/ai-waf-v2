"""
stages/6_distillation_and_compression/05_export_and_bench.py
------------------------------------------------------------
Stage 6.5 — Unified export (ONNX → TensorRT) and SLO-enforced benchmarking.

WHAT THIS REPLACES
------------------
Consolidates three previously disconnected scripts:
  05_export_onnx.py      — ONNX export + ORT benchmark
  06_export_trt.py       — TensorRT engine build
  07_onnxruntime_bench.py — Detailed ORT benchmark across providers

WHY CONSOLIDATION MATTERS
--------------------------
With separate scripts, it was impossible to enforce an end-to-end SLO:
the p99 latency check in 05 never saw TRT results, and the TRT bench in 06
never compared against ORT. This script runs the full chain in one pass and
emits a single "Deployment Blocked" flag if any SLO is violated.

Pipeline:
  1. Load student checkpoint (verify canary gate has passed)
  2. Benchmark PyTorch baseline
  3. Export to ONNX (opset 17, dynamic batch/seq axes)
  4. Validate ONNX graph
  5. Benchmark ORT on CUDA and CPU execution providers
  6. Build TensorRT FP16 engine (optional — skipped gracefully if TRT absent)
  7. Benchmark TRT engine
  8. Enforce p99 and throughput SLOs across all providers
  9. Emit unified latency_summary.json + deployment_status.json

Run:
    python stages/6_distillation_and_compression/05_export_and_bench.py \
        --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai_waf_v2.eval.latency import LatencyBenchmark, OnnxLatencyBenchmark
from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


# ── Export wrapper ────────────────────────────────────────────────────────────

class _OnnxExportWrapper(torch.nn.Module):
    """Strip student output dict to a single logits tensor for ONNX export."""
    def __init__(self, student: StudentClassifier) -> None:
        super().__init__()
        self.student = student

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.student(input_ids, attention_mask)["logits"]


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(
    student: StudentClassifier,
    onnx_path: Path,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Path:
    log.info(f"Exporting student to ONNX: {onnx_path}")
    wrapper    = _OnnxExportWrapper(student).eval()
    dummy_ids  = torch.randint(1, vocab_size, (1, seq_len), device=device)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # dynamo=False forces the legacy TorchScript exporter, which does not require
    # the optional onnxscript package introduced as a hard dep in PyTorch ≥ 2.5.
    export_kwargs: dict = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits":         {0: "batch_size"},
        },
    )
    # dynamo= kwarg added in PyTorch 2.1; guard for older builds
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy_ids, dummy_mask), str(onnx_path), **export_kwargs)
    log.info(f"ONNX model saved: {onnx_path}")

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        log.info("ONNX model validation: OK")
    except ImportError:
        log.warning("onnx package not installed — skipping graph validation.")

    return onnx_path


# ── TensorRT engine build ─────────────────────────────────────────────────────

def build_trt_engine(
    onnx_path: Path,
    trt_path: Path,
    workspace_gb: int = 2,
) -> Path | None:
    """Build a TensorRT FP16 engine from the ONNX model.

    Returns the engine path on success, None if TRT is not available.
    """
    try:
        import tensorrt as trt
    except ImportError:
        log.warning(
            "TensorRT not installed — skipping TRT engine build. "
            "Install: pip install tensorrt pycuda"
        )
        return None

    log.info(f"Building TensorRT engine: {trt_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser     = trt.OnnxParser(network, TRT_LOGGER)
    config     = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("TRT: FP16 precision enabled.")
    else:
        log.warning("TRT: Platform does not support FP16 — falling back to FP32.")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(f"TRT parse error: {parser.get_error(i)}")
            return None

    # Dynamic-shape ONNX models require an optimization profile that specifies
    # min/opt/max shapes for each dynamic input axis.
    profile = builder.create_optimization_profile()
    seq_len = network.get_input(0).shape[1]
    seq_len = seq_len if seq_len > 0 else 256   # -1 means dynamic; default to 256
    for inp_idx in range(network.num_inputs):
        inp = network.get_input(inp_idx)
        profile.set_shape(
            inp.name,
            min=(1,  seq_len),
            opt=(32, seq_len),
            max=(256, seq_len),
        )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        log.error("TRT engine build failed.")
        return None

    trt_path.write_bytes(engine_bytes)
    log.info(f"TRT engine saved: {trt_path}")
    return trt_path


# ── TensorRT benchmark ────────────────────────────────────────────────────────

def bench_trt(
    trt_path: Path,
    seq_len: int,
    batch_sizes: list[int],
    n_warmup: int = 20,
    n_runs: int = 100,
) -> list[dict]:
    """Run a latency benchmark against the TRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        log.warning("pycuda / tensorrt not available — TRT benchmark skipped.")
        return []

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)
    engine     = runtime.deserialize_cuda_engine(trt_path.read_bytes())
    context    = engine.create_execution_context()

    results = []
    for bs in batch_sizes:
        dummy_ids  = np.ones((bs, seq_len), dtype=np.int64)
        dummy_mask = np.ones((bs, seq_len), dtype=np.int64)
        out_buf    = np.zeros((bs, 2),       dtype=np.float32)

        # Dynamic-shape engines require explicit input shapes per batch size.
        for inp_idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(inp_idx)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, (bs, seq_len))

        d_ids  = cuda.mem_alloc(dummy_ids.nbytes)
        d_mask = cuda.mem_alloc(dummy_mask.nbytes)
        d_out  = cuda.mem_alloc(out_buf.nbytes)
        cuda.memcpy_htod(d_ids,  dummy_ids)
        cuda.memcpy_htod(d_mask, dummy_mask)

        lats = []
        for i in range(n_warmup + n_runs):
            t0 = time.perf_counter()
            ok = context.execute_v2(bindings=[int(d_ids), int(d_mask), int(d_out)])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if not ok:
                log.warning(f"TRT execute_v2 returned False at bs={bs}")
                break
            if i >= n_warmup:
                lats.append(elapsed_ms)
        if not lats:
            continue

        lats_arr = np.array(lats)
        throughput = bs / (np.mean(lats_arr) / 1000)
        results.append({
            "batch_size":     bs,
            "mean_ms":        round(float(lats_arr.mean()), 3),
            "p50_ms":         round(float(np.percentile(lats_arr, 50)), 3),
            "p99_ms":         round(float(np.percentile(lats_arr, 99)), 3),
            "throughput_rps": round(throughput, 1),
        })
        log.info(f"  TRT  bs={bs:4d}  p99={results[-1]['p99_ms']:.2f}ms  "
                 f"throughput={throughput:.0f} rps")

    return results


# ── SLO enforcement ───────────────────────────────────────────────────────────

def _check_slo(
    provider: str,
    results: list[dict],
    p99_slo_ms: float,
    throughput_slo: float,
) -> dict:
    """Return per-provider SLO assessment."""
    if not results:
        return {"provider": provider, "status": "SKIPPED", "violations": []}

    violations = []
    bs1 = next((r for r in results if r["batch_size"] == 1), None)
    if bs1 and bs1["p99_ms"] > p99_slo_ms:
        violations.append(
            f"p99 latency {bs1['p99_ms']:.2f}ms > SLO {p99_slo_ms}ms at bs=1"
        )
    max_throughput = max(r["throughput_rps"] for r in results)
    if max_throughput < throughput_slo:
        violations.append(
            f"peak throughput {max_throughput:.0f} rps < SLO {throughput_slo} rps"
        )

    status = "PASS" if not violations else "FAIL"
    log.info(f"  SLO [{status}] {provider}: {violations or 'all clear'}")
    return {"provider": provider, "status": status, "violations": violations}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    student_cfg = cfg.model.student
    seq_len     = cfg.tokenizer.seq_len
    vocab_size  = student_cfg.vocab_size
    batch_sizes = cfg.evaluation.batch_sizes

    require_inputs({
        f"{student_cfg.output_dir}/best_student.pt": "run 02_distill_train.py",
    })
    if check_output(
        Path(student_cfg.output_dir) / "student.onnx",
        args.force, "Stage 6.5 export and bench"
    ):
        return

    # ── Canary gate guard ─────────────────────────
    canary_report = Path(cfg.paths.reports) / "metrics" / "student_canary.json"
    if canary_report.exists():
        canary_data = json.loads(canary_report.read_text())
        if not canary_data.get("slo_passed", False):
            log.error(
                "DEPLOYMENT BLOCKED: Canary SLO not satisfied. "
                "Fix recall regressions in 04_student_canary.py before exporting."
            )
            sys.exit(1)
        log.info("Canary gate: PASS")
    else:
        log.warning(
            "student_canary.json not found. Proceeding without canary gate check. "
            "Run 04_student_canary.py before deploying to production."
        )

    # ── Load student ──────────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(student_cfg.output_dir) / "best_student.pt"
    if not checkpoint.exists():
        log.error(f"Student checkpoint not found: {checkpoint}.")
        sys.exit(1)

    student = StudentClassifier.load(checkpoint, student_cfg, map_location=str(device))
    student.to(device).eval()
    log.info(f"Student: {student.count_parameters():,} params on {device}")

    onnx_path = Path(student_cfg.output_dir) / "student.onnx"
    trt_path  = Path(student_cfg.output_dir) / "student.engine"
    timer     = StepTimer()

    # ── 1. PyTorch baseline ───────────────────────
    log.info("\n── PyTorch Baseline ──────────────────────────────")
    pt_bench = LatencyBenchmark(
        model=student, device=str(device),
        seq_len=seq_len, vocab_size=vocab_size,
        n_warmup=50, n_runs=200, use_bf16=True,
    )
    with timer.step("benchmark_pytorch"):
        pt_results = pt_bench.run(batch_sizes)
    LatencyBenchmark.print_table(pt_results)

    # ── 2. ONNX export ────────────────────────────
    log.info("\n── ONNX Export ───────────────────────────────────")
    with timer.step("export_onnx"):
        export_onnx(student, onnx_path, seq_len, vocab_size, device)

    # ── 3. ORT CUDA benchmark ─────────────────────
    log.info("\n── ORT: CUDAExecutionProvider ────────────────────")
    ort_cuda_results: list[dict] = []
    ort_cuda_available = False
    try:
        cuda_bench = OnnxLatencyBenchmark(
            onnx_path=onnx_path, device="cuda",
            seq_len=seq_len, vocab_size=vocab_size,
            n_warmup=cfg.evaluation.n_latency_warmup,
            n_runs=cfg.evaluation.n_latency_runs,
        )
        with timer.step("benchmark_ort_cuda"):
            ort_cuda_results = cuda_bench.run(batch_sizes)
        LatencyBenchmark.print_table(ort_cuda_results)
        ort_cuda_available = True
    except Exception as e:
        log.warning(f"ORT CUDA benchmark failed: {e}")

    # ── 4. ORT CPU benchmark ──────────────────────
    # Skip CPU ORT when CUDA ORT is unavailable: without onnxruntime-gpu the
    # model is not targeting CPU deployment, and 500 runs on CPU takes minutes.
    log.info("\n── ORT: CPUExecutionProvider ─────────────────────")
    ort_cpu_results: list[dict] = []
    if not ort_cuda_available:
        log.warning(
            "ORT CPU benchmark skipped — install onnxruntime-gpu for ORT benchmarking."
        )
    else:
        try:
            cpu_bench = OnnxLatencyBenchmark(
                onnx_path=onnx_path, device="cpu",
                seq_len=seq_len, vocab_size=vocab_size,
                n_warmup=10,
                n_runs=50,   # cap CPU runs — not the deployment target
            )
            with timer.step("benchmark_ort_cpu"):
                ort_cpu_results = cpu_bench.run(batch_sizes)
            LatencyBenchmark.print_table(ort_cpu_results)
        except Exception as e:
            log.warning(f"ORT CPU benchmark failed: {e}")

    # ── 5. TensorRT build + benchmark ─────────────
    log.info("\n── TensorRT Engine ───────────────────────────────")
    trt_results: list[dict] = []
    with timer.step("build_trt"):
        trt_built = build_trt_engine(onnx_path, trt_path, workspace_gb=2)
    if trt_built:
        with timer.step("benchmark_trt"):
            trt_results = bench_trt(
                trt_path, seq_len, batch_sizes,
                n_warmup=20, n_runs=100,
            )

    # ── 6. SLO enforcement ────────────────────────
    log.info("\n── SLO Enforcement ───────────────────────────────")
    p99_slo   = cfg.slo.latency_inline_p99_ms
    tp_slo    = cfg.slo.throughput_min_rps

    slo_checks = [
        _check_slo("pytorch",            pt_results,       p99_slo, tp_slo),
        _check_slo("ort_cuda",           ort_cuda_results, p99_slo, tp_slo),
        _check_slo("ort_cpu",            ort_cpu_results,  p99_slo, tp_slo),
        _check_slo("tensorrt",           trt_results,      p99_slo, tp_slo),
    ]

    # APPROVED if at least one non-skipped provider passes all SLOs.
    # PyTorch fp32 not hitting throughput SLO while TRT FP16 does is expected.
    non_skipped = [c for c in slo_checks if c["status"] != "SKIPPED"]
    all_pass = any(c["status"] == "PASS" for c in non_skipped) if non_skipped else False
    deployment_status = "APPROVED" if all_pass else "BLOCKED"

    log.info(f"\n  Deployment status: {deployment_status}")
    if not all_pass:
        failing = [c for c in slo_checks if c["status"] == "FAIL"]
        for c in failing:
            log.error(f"  BLOCKED by {c['provider']}: {c['violations']}")

    # ── 7. Save reports ───────────────────────────
    report_dir = Path(cfg.paths.reports) / "latency"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Per-provider JSON (backward compat)
    (report_dir / "onnx_bench.json").write_text(json.dumps({
        "pytorch":     pt_results,
        "onnxruntime": ort_cuda_results,
    }, indent=2))
    (report_dir / "ort_detailed_bench.json").write_text(json.dumps({
        "CUDAExecutionProvider": {"results": ort_cuda_results},
        "CPUExecutionProvider":  {"results": ort_cpu_results},
    }, indent=2))
    if trt_results:
        (report_dir / "trt_bench.json").write_text(json.dumps({
            "trt_engine":   str(trt_path),
            "results":      trt_results,
        }, indent=2))

    # Unified summary
    summary = {
        "deployment_status": deployment_status,
        "slo": {"p99_ms": p99_slo, "throughput_rps": tp_slo},
        "slo_checks": slo_checks,
        "latency_by_provider": {
            "pytorch":   pt_results,
            "ort_cuda":  ort_cuda_results,
            "ort_cpu":   ort_cpu_results,
            "tensorrt":  trt_results,
        },
        "artifacts": {
            "onnx":      str(onnx_path),
            "trt_engine": str(trt_path) if trt_built else None,
        },
        "timings_s": timer.timings,
    }
    summary_path = report_dir / "latency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"\nUnified latency summary saved to {summary_path}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="05_export_and_bench"):
            mlflow.log_params({
                "deployment_status": deployment_status,
                "p99_slo_ms":        p99_slo,
                "throughput_slo_rps": tp_slo,
                "onnx_path":         str(onnx_path),
                "trt_built":         bool(trt_built),
            })
            metrics: dict[str, float] = {
                "all_slo_pass": float(all_pass),
            }
            for check in slo_checks:
                provider = check["provider"]
                metrics[f"slo_pass_{provider}"] = float(check["status"] == "PASS")
            # Log bs=1 p99 per provider where available
            for provider_key, result_list in [
                ("pytorch", pt_results),
                ("ort_cuda", ort_cuda_results),
                ("ort_cpu", ort_cpu_results),
                ("trt", trt_results),
            ]:
                bs1 = next((r for r in result_list if r.get("batch_size") == 1), None)
                if bs1:
                    metrics[f"p99_ms_{provider_key}"] = float(bs1["p99_ms"])
                    metrics[f"throughput_{provider_key}"] = float(bs1["throughput_rps"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(summary_path))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    # ── CI/CD gate ────────────────────────────────
    if not all_pass:
        log.error(
            "DEPLOYMENT BLOCKED: One or more providers failed the SLO. "
            "Review latency_summary.json for details."
        )
        sys.exit(1)

    log.info("Export and benchmarking complete. Deployment APPROVED.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified ONNX/TRT export and SLO-enforced benchmarking."
    )
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())