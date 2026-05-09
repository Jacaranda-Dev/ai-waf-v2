"""Stage 5.3 — Post-training quantization (PTQ) of the teacher 99M model."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from ai_waf_v2.eval.latency import LatencyBenchmark
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"
    if not ckpt.exists(): log.error(f"Teacher checkpoint not found: {ckpt}"); return
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    # Baseline fp16 latency
    bench_fp16 = LatencyBenchmark(model=model, device=str(device),
                                   seq_len=cfg.tokenizer.seq_len,
                                   vocab_size=cfg.model.track_b_99m.vocab_size,
                                   n_warmup=20, n_runs=100, use_bf16=True)
    results_fp16 = bench_fp16.run([1, 64])
    log.info("fp16/bf16 baseline:")
    LatencyBenchmark.print_table(results_fp16)
    # PTQ with bitsandbytes (INT8)
    results_int8 = []
    try:
        import bitsandbytes as bnb
        from ai_waf_v2.models.student import StudentClassifier
        # Re-load and quantize
        model_int8 = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location="cpu")
        # Replace Linear with Linear8bitLt
        def _replace(m):
            for name, child in m.named_children():
                if isinstance(child, torch.nn.Linear):
                    new = bnb.nn.Linear8bitLt(child.in_features, child.out_features,
                                               bias=child.bias is not None,
                                               has_fp16_weights=False, threshold=6.0)
                    new.weight = child.weight
                    if child.bias is not None: new.bias = child.bias
                    setattr(m, name, new)
                else: _replace(child)
        _replace(model_int8.encoder.layers)
        model_int8.to(device).eval()
        bench_int8 = LatencyBenchmark(model=model_int8, device=str(device),
                                       seq_len=cfg.tokenizer.seq_len,
                                       vocab_size=cfg.model.track_b_99m.vocab_size,
                                       n_warmup=20, n_runs=100, use_bf16=False)
        results_int8 = bench_int8.run([1, 64])
        log.info("INT8 PTQ (teacher):")
        LatencyBenchmark.print_table(results_int8)
    except ImportError:
        log.warning("bitsandbytes not installed — skipping INT8 PTQ")
    out = Path(cfg.paths.reports)/"latency"/"ptq_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fp16": results_fp16, "int8_ptq": results_int8}, indent=2))
    log.info(f"PTQ comparison saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
