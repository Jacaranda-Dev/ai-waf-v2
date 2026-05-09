"""
stages/6_distillation_and_compression/05_export_onnx.py
----------------------------------
Stage 5.4 — Export the student model to ONNX and benchmark with
onnxruntime CUDA execution provider.

Produces:
    models/student/student.onnx
    reports/latency/onnx_bench.json

Run:
    python stages/6_distillation_and_compression/05_export_onnx.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai_waf_v2.eval.latency import LatencyBenchmark, OnnxLatencyBenchmark
from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# Wrapped student that returns only logits tensor (ONNX needs single output)
class _ExportWrapper(torch.nn.Module):
    def __init__(self, student: StudentClassifier) -> None:
        super().__init__()
        self.student = student

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.student(input_ids, attention_mask)
        return out["logits"]


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    student_cfg = cfg.model.student
    seq_len     = cfg.tokenizer.seq_len
    vocab_size  = student_cfg.vocab_size

    checkpoint = Path(student_cfg.output_dir) / "best_student.pt"
    if not checkpoint.exists():
        log.error(f"Student checkpoint not found: {checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = StudentClassifier.load(checkpoint, student_cfg, map_location=str(device))
    student.to(device).eval()
    log.info(f"Student loaded: {student.count_parameters():,} params")

    # ── PyTorch baseline latency ──────────────────
    log.info("Benchmarking PyTorch student inference...")
    pt_bench = LatencyBenchmark(
        model=student,
        device=str(device),
        seq_len=seq_len,
        vocab_size=vocab_size,
        n_warmup=50,
        n_runs=200,
        use_bf16=True,
    )
    pt_results = pt_bench.run(batch_sizes=[1, 8, 32, 64, 256])
    LatencyBenchmark.print_table(pt_results)
    slo_ok = pt_bench.check_slo(pt_results,
        inline_p99_ms=cfg.slo.latency_inline_p99_ms,
        throughput_min=cfg.slo.throughput_min_rps,
    )
    log.info(f"SLO check: {slo_ok}")

    # ── ONNX export ───────────────────────────────
    onnx_path = Path(student_cfg.output_dir) / "student.onnx"
    log.info(f"Exporting to ONNX: {onnx_path}")

    wrapper = _ExportWrapper(student)
    wrapper.eval()

    dummy_ids  = torch.randint(1, vocab_size, (1, seq_len), device=device)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(onnx_path),
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
    log.info(f"ONNX model saved: {onnx_path}")

    # Verify ONNX model is valid
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        log.info("ONNX model validation: OK")
    except ImportError:
        log.warning("onnx package not installed — skipping validation")

    # ── ONNX Runtime benchmark ────────────────────
    log.info("Benchmarking ONNX Runtime...")
    try:
        ort_bench = OnnxLatencyBenchmark(
            onnx_path=onnx_path,
            device=str(device),
            seq_len=seq_len,
            vocab_size=vocab_size,
            n_warmup=50,
            n_runs=200,
        )
        ort_results = ort_bench.run(batch_sizes=[1, 8, 32, 64, 256])
        LatencyBenchmark.print_table(ort_results)
    except Exception as e:
        log.warning(f"ONNX Runtime benchmark failed: {e}")
        ort_results = []

    # ── Save combined report ──────────────────────
    report_dir = Path(cfg.paths.reports) / "latency"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "pytorch": pt_results,
        "onnxruntime": ort_results,
        "slo_check": slo_ok,
    }
    out_path = report_dir / "onnx_bench.json"
    out_path.write_text(json.dumps(report, indent=2))
    log.info(f"Report saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())