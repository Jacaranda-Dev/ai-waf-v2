"""
stages/6_distillation_and_compression/01_student_arch.py
---------------------------------------------------------
Stage 6.1 — Print student architecture summary, parameter breakdown,
and validate vocab/dimension consistency against the teacher config.

Changes from original:
  - Added vocab_size consistency check (teacher vs student).
  - Added head-layer breakdown to the saved JSON.
  - Added compression ratio guard: warns if ratio is below 5× or above 50×.
  - Improved structured logging with cleaner formatting.

Run:
    python stages/6_distillation_and_compression/01_student_arch.py \
        --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

_COMPRESSION_WARN_LOW  = 5.0
_COMPRESSION_WARN_HIGH = 50.0


def _validate_config_consistency(scfg, tcfg) -> None:
    """Warn on obvious mismatches between student and teacher configs."""
    if scfg.vocab_size != tcfg.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: student={scfg.vocab_size}, teacher={tcfg.vocab_size}. "
            "Both models must share the same tokenizer vocabulary to ensure valid "
            "soft-target distributions during distillation."
        )
    log.info(f"Config consistency check passed: vocab_size={scfg.vocab_size}")


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg  = load_config(args.config)
    scfg = cfg.model.student
    tcfg = cfg.model.track_b_99m

    require_inputs({})
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "student_arch.json",
        args.force, "Stage 6.1 student arch check"
    ):
        return

    timer = StepTimer()

    # ── Consistency gate ──────────────────────────
    _validate_config_consistency(scfg, tcfg)

    # ── Build student (CPU only — architecture check) ─
    with timer.step("build_student"):
        student = StudentClassifier.from_config(scfg, teacher_d_model=tcfg.d_model)
        n_s = student.count_parameters()
        n_t = 99_000_000  # canonical teacher size

        breakdown = student.encoder.parameter_breakdown()
        head_n    = sum(p.numel() for p in student.head.parameters())

        compression = n_t / max(n_s, 1)
        vram_fp16   = n_s * 2 / 1e6   # 2 bytes per param
        vram_int8   = n_s * 1 / 1e6   # 1 byte per param

    # ── Report ────────────────────────────────────
    log.info("=" * 60)
    log.info("Student Architecture Summary")
    log.info("=" * 60)
    log.info(f"  d_model       : {scfg.d_model}")
    log.info(f"  n_layers      : {scfg.n_layers}")
    log.info(f"  n_heads       : {scfg.n_heads}")
    log.info(f"  d_ff          : {scfg.d_ff}")
    log.info(f"  vocab_size    : {scfg.vocab_size}")
    log.info(f"  quantization  : {scfg.quantization}")
    log.info("-" * 60)
    log.info(f"  Embeddings    : {breakdown['embeddings']:>12,}")
    log.info(f"  Attention     : {breakdown['attention']:>12,}")
    log.info(f"  FFN           : {breakdown['ffn']:>12,}")
    log.info(f"  Head          : {head_n:>12,}")
    log.info(f"  Total params  : {n_s:>12,}")
    log.info("-" * 60)
    log.info(f"  Compression   : {compression:.1f}× vs teacher 99M")
    log.info(f"  VRAM fp16     : {vram_fp16:.0f} MB")
    log.info(f"  VRAM int8     : {vram_int8:.0f} MB")
    log.info("=" * 60)

    # ── Compression ratio sanity check ────────────
    if compression < _COMPRESSION_WARN_LOW:
        log.warning(
            f"Compression ratio {compression:.1f}× is lower than {_COMPRESSION_WARN_LOW}×. "
            "The student may be too large for meaningful latency gains."
        )
    elif compression > _COMPRESSION_WARN_HIGH:
        log.warning(
            f"Compression ratio {compression:.1f}× exceeds {_COMPRESSION_WARN_HIGH}×. "
            "Aggressive compression at this scale may cause significant accuracy loss."
        )

    # ── Save JSON ─────────────────────────────────
    result = {
        "n_params":             n_s,
        "compression_vs_99m":  round(compression, 1),
        "arch": {
            "d_model":    scfg.d_model,
            "n_layers":   scfg.n_layers,
            "n_heads":    scfg.n_heads,
            "d_ff":       scfg.d_ff,
            "vocab_size": scfg.vocab_size,
        },
        "vram_fp16_mb": round(vram_fp16, 1),
        "vram_int8_mb": round(vram_int8, 1),
        "breakdown": {
            **breakdown,
            "head": head_n,
        },
        "quantization": scfg.quantization,
        "warnings": {
            "compression_low":  compression < _COMPRESSION_WARN_LOW,
            "compression_high": compression > _COMPRESSION_WARN_HIGH,
        },
        "timings_s": timer.timings,
    }

    out = Path(cfg.paths.reports) / "metrics" / "student_arch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Student arch summary saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_student_arch"):
            mlflow.log_params({
                "d_model":         scfg.d_model,
                "n_layers":        scfg.n_layers,
                "n_heads":         scfg.n_heads,
                "d_ff":            scfg.d_ff,
                "vocab_size":      scfg.vocab_size,
                "quantization":    scfg.quantization,
                "teacher_d_model": tcfg.d_model,
            })
            log_metrics_dict({
                "n_params":            float(n_s),
                "compression_vs_99m":  float(compression),
                "vram_fp16_mb":        float(vram_fp16),
                "vram_int8_mb":        float(vram_int8),
                "compression_warn_low":  float(compression < _COMPRESSION_WARN_LOW),
                "compression_warn_high": float(compression > _COMPRESSION_WARN_HIGH),
            })
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print student architecture summary.")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())