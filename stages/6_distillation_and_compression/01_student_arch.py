"""Stage 5.1 — Print student architecture summary and parameter breakdown."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    scfg    = cfg.model.student; tcfg = cfg.model.track_b_99m
    student = StudentClassifier.from_config(scfg, teacher_d_model=tcfg.d_model)
    n_s = student.count_parameters()
    n_t = 99_000_000  # approximate teacher
    breakdown = student.encoder.parameter_breakdown()
    head_n    = sum(p.numel() for p in student.head.parameters())
    log.info(f"Student architecture:")
    log.info(f"  d_model={scfg.d_model}  n_layers={scfg.n_layers}  n_heads={scfg.n_heads}  d_ff={scfg.d_ff}")
    log.info(f"  Total params : {n_s:,}")
    log.info(f"  Compression  : {n_t/n_s:.1f}× vs teacher 99M")
    log.info(f"  Embeddings   : {breakdown[\'embeddings\']:,}")
    log.info(f"  Attention    : {breakdown[\'attention\']:,}")
    log.info(f"  FFN          : {breakdown[\'ffn\']:,}")
    log.info(f"  Head         : {head_n:,}")
    log.info(f"  Quantization : {scfg.quantization}")
    vram_fp16 = n_s * 2 / 1e6
    vram_int8 = n_s * 1 / 1e6
    log.info(f"  VRAM fp16    : {vram_fp16:.0f} MB")
    log.info(f"  VRAM int8    : {vram_int8:.0f} MB")
    result = {"n_params": n_s, "compression_vs_99m": round(n_t/n_s,1),
              "arch": {"d_model":scfg.d_model,"n_layers":scfg.n_layers,"n_heads":scfg.n_heads},
              "vram_fp16_mb": round(vram_fp16,1), "vram_int8_mb": round(vram_int8,1),
              "breakdown": breakdown}
    out = Path(cfg.paths.reports)/"metrics"/"student_arch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Student arch summary saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
